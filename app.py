import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Market Intelligence Lab v0.2", layout="wide")
st.title("Market Intelligence Lab v0.2")
st.caption("Private research prototype: market anomalies + world events. No live trading.")

SYMBOLS = ["SPY", "QQQ", "XLE", "GLD", "TLT"]

def make_demo_data(symbol="SPY", n=300):
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    start = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=n)
    ts = pd.date_range(start, periods=n, freq="min")
    returns = rng.normal(0, 0.00055, n)
    volume = rng.lognormal(11.2, 0.42, n).astype(int)
    returns[-25:-18] += np.linspace(-0.001, -0.004, 7)
    volume[-25:-18] *= 7
    close = 500 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "timestamp": ts, "open": close * .9998, "high": close * 1.0006,
        "low": close * .9994, "close": close, "volume": volume
    })

def alpaca_bars(symbol, start, end):
    key = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY", ""))
    secret = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))
    if not key or not secret:
        raise RuntimeError("Alpaca API keys are not configured yet.")
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"timeframe":"1Min","start":start.isoformat(),"end":end.isoformat(),
              "feed":"iex","limit":10000,"sort":"asc"}
    rows = []
    while True:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        rows.extend(js.get("bars", []))
        token = js.get("next_page_token")
        if not token: break
        params["page_token"] = token
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp","open","high","low","close","volume"]]

def compute_features(df):
    x = df.sort_values("timestamp").copy()
    x["return_1m"] = x["close"].pct_change()
    x["volume_z"] = (x["volume"]-x["volume"].rolling(60,min_periods=20).mean()) / x["volume"].rolling(60,min_periods=20).std()
    x["return_z"] = (x["return_1m"]-x["return_1m"].rolling(60,min_periods=20).mean()) / x["return_1m"].rolling(60,min_periods=20).std()
    vol = x["return_1m"].rolling(15,min_periods=5).std()
    x["volatility_z"] = (vol-vol.rolling(60,min_periods=20).mean()) / vol.rolling(60,min_periods=20).std()
    raw = .45*x["volume_z"].fillna(0).clip(lower=0) + .35*x["return_z"].abs().fillna(0) + .20*x["volatility_z"].fillna(0).clip(lower=0)
    x["anomaly_score"] = (100*(1-np.exp(-raw/3))).clip(0,100)
    return x

def load_events():
    path = Path(__file__).with_name("events_sample.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

st.sidebar.header("Control panel")
mode = st.sidebar.radio("Data source", ["Demo", "Alpaca real data"])
symbol = st.sidebar.selectbox("Asset", SYMBOLS)
threshold = st.sidebar.slider("Alert threshold", 50, 95, 75)

if mode == "Demo":
    df = make_demo_data(symbol)
else:
    end = datetime.now(timezone.utc)-timedelta(minutes=16)
    start = end-timedelta(days=3)
    try:
        df = alpaca_bars(symbol,start,end)
        if df.empty: df = make_demo_data(symbol)
    except Exception as e:
        st.error(f"Real-data connection not ready: {e}")
        df = make_demo_data(symbol)

feat = compute_features(df)
events = load_events()
tabs = st.tabs(["Market Replay","Anomaly Detector","Event Intelligence","Historical Events","Results"])

with tabs[0]:
    st.subheader(f"{symbol} — Market Replay")
    st.line_chart(feat.set_index("timestamp")[["close"]])
    st.bar_chart(feat.set_index("timestamp")[["volume"]])

with tabs[1]:
    st.subheader("Pre-event Anomaly Detector")
    valid = feat.dropna(subset=["anomaly_score"])
    latest = valid.iloc[-1] if not valid.empty else feat.iloc[-1]
    a,b,c,d = st.columns(4)
    a.metric("Anomaly score", f"{latest.get('anomaly_score',0):.1f}/100")
    b.metric("Volume z", f"{latest.get('volume_z',0):.2f}")
    c.metric("Return z", f"{latest.get('return_z',0):.2f}")
    d.metric("Alert threshold", threshold)
    st.line_chart(feat.set_index("timestamp")[["anomaly_score"]])
    st.dataframe(feat[feat["anomaly_score"]>=threshold].tail(30), use_container_width=True)

with tabs[2]:
    st.subheader("Event Intelligence")
    st.markdown("""
This module will connect world events with affected markets.

**Sensors planned:** political/government statements, geopolitics, CEO/company events,
Fed/macro/OPEC releases, breaking events, prediction markets and cross-market anomalies.

**EVENT → MARKET:** What could move, in which direction, and is it already priced?

**MARKET → POSSIBLE EVENT:** Are unexplained simultaneous anomalies indicating that
something important may be happening before public confirmation?
""")
    st.info("Live news ingestion will be added after establishing the historical baseline.")

with tabs[3]:
    st.subheader("Historical Events")
    st.dataframe(events, use_container_width=True)
    st.warning("Seed timestamps must be verified before being used in final statistics.")

with tabs[4]:
    st.subheader("Research Results")
    st.markdown("""
Compare T−60 / T−30 / T−15 / T−5 / T−1 before verified events with matched normal windows.

We will measure hit rate, false positives, precision, recall, predictive lift and forward
returns at +5m / +15m / +30m / +60m / +1 day.

**First target:** Can the system distinguish the period immediately before a major event
from an ordinary period better than chance?
""")

st.divider()
st.caption("Research only • v0.2 • No brokerage connection • No automatic orders")
