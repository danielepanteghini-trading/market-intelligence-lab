
import os
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Anomaly Lab v0.1", layout="wide")

st.title("Anomaly Lab v0.1")
st.caption("Research prototype: market anomalies before major events. No live trading.")

SYMBOLS = ["SPY", "QQQ", "XLE", "GLD", "TLT"]

def zscore(series):
    s = pd.Series(series, dtype=float)
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std

def make_demo_data(symbol="SPY", n=240):
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    start = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=n)
    ts = pd.date_range(start, periods=n, freq="min")
    ret = rng.normal(0, 0.00055, n)
    price = 500 * np.exp(np.cumsum(ret))
    volume = rng.lognormal(mean=11.2, sigma=0.42, size=n).astype(int)
    # inject a visible anomaly near the end
    volume[-18:-12] *= 7
    ret[-18:-12] += np.linspace(-0.0015, -0.004, 6)
    price = 500 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "timestamp": ts,
        "open": price * (1 - 0.0002),
        "high": price * (1 + 0.0006),
        "low": price * (1 - 0.0006),
        "close": price,
        "volume": volume,
    })

def alpaca_bars(symbol, start, end, timeframe="1Min"):
    key = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY", ""))
    secret = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))
    if not key or not secret:
        raise RuntimeError("Alpaca API keys not configured.")
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    params = {
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "feed": "iex",
        "limit": 10000,
        "sort": "asc",
    }
    rows = []
    while True:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        rows.extend(js.get("bars", []))
        token = js.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp","open","high","low","close","volume"]]

def compute_features(df):
    x = df.copy().sort_values("timestamp")
    x["return_1m"] = x["close"].pct_change()
    x["abs_return_1m"] = x["return_1m"].abs()
    x["volume_z_60"] = (
        x["volume"] - x["volume"].rolling(60, min_periods=20).mean()
    ) / x["volume"].rolling(60, min_periods=20).std()
    x["return_z_60"] = (
        x["return_1m"] - x["return_1m"].rolling(60, min_periods=20).mean()
    ) / x["return_1m"].rolling(60, min_periods=20).std()
    x["volatility_15"] = x["return_1m"].rolling(15, min_periods=5).std()
    x["volatility_z_60"] = (
        x["volatility_15"] - x["volatility_15"].rolling(60, min_periods=20).mean()
    ) / x["volatility_15"].rolling(60, min_periods=20).std()

    # Initial transparent anomaly score. We will calibrate this statistically later.
    vz = x["volume_z_60"].fillna(0).clip(lower=0)
    rz = x["return_z_60"].abs().fillna(0)
    zz = x["volatility_z_60"].fillna(0).clip(lower=0)
    raw = 0.45*vz + 0.35*rz + 0.20*zz
    x["anomaly_score"] = (100 * (1 - np.exp(-raw/3))).clip(0,100)
    return x

st.sidebar.header("Data source")
mode = st.sidebar.radio("Mode", ["Demo", "Alpaca real data"], index=0)
symbol = st.sidebar.selectbox("Symbol", SYMBOLS)

if mode == "Demo":
    df = make_demo_data(symbol)
else:
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    start = end - timedelta(days=3)
    try:
        df = alpaca_bars(symbol, start, end)
        if df.empty:
            st.warning("No bars returned. Falling back to demo.")
            df = make_demo_data(symbol)
    except Exception as e:
        st.error(f"Alpaca error: {e}")
        st.info("Running demo data instead.")
        df = make_demo_data(symbol)

feat = compute_features(df)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Market Replay", "Anomaly Detector", "Historical Events", "Results"]
)

with tab1:
    st.subheader(f"{symbol} — replay")
    st.line_chart(feat.set_index("timestamp")[["close"]])
    st.bar_chart(feat.set_index("timestamp")[["volume"]])

with tab2:
    st.subheader("Anomaly score")
    c1, c2, c3 = st.columns(3)
    latest = feat.dropna(subset=["anomaly_score"]).iloc[-1]
    c1.metric("Current anomaly score", f"{latest['anomaly_score']:.1f}/100")
    c2.metric("Volume z-score", f"{latest['volume_z_60']:.2f}")
    c3.metric("Return z-score", f"{latest['return_z_60']:.2f}")
    st.line_chart(feat.set_index("timestamp")[["anomaly_score"]])
    alerts = feat[feat["anomaly_score"] >= 75][
        ["timestamp","close","volume","volume_z_60","return_z_60","anomaly_score"]
    ].tail(20)
    st.dataframe(alerts, use_container_width=True)

with tab3:
    st.subheader("Event catalog")
    events = pd.read_csv(Path(__file__).with_name("events_sample.csv"))
    st.dataframe(events, use_container_width=True)
    st.caption(
        "These are seed events for the research workflow. Exact timestamps must be "
        "verified from primary/credible sources before statistical use."
    )

with tab4:
    st.subheader("Experiment design")
    st.markdown("""
    **Primary test**

    Compare anomaly scores in windows **T−60, T−30, T−15, T−5, T−1** before
    verified market-moving events against randomly sampled control windows matched
    by symbol, weekday and time-of-day.

    **Metrics**
    - Event-window hit rate at score thresholds 70 / 80 / 90
    - False positive rate on control windows
    - Precision, recall and ROC-AUC
    - Forward returns at +5m / +15m / +30m / +60m / +1d
    - Results after simulated spread/slippage
    """)
    st.info("v0.1 deliberately does not place trades.")
