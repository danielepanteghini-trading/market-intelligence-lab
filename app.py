import os
from datetime import datetime, timedelta, timezone
from itertools import product

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Market Intelligence Lab v0.7", layout="wide")
st.title("Market Intelligence Lab v0.7")
st.caption("Technical Scanner + Edge Factory • research only • no automatic orders")

@st.cache_data(ttl=900, show_spinner=False)
def getbars(sym, days, tf):
    key = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY", ""))
    secret = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY", ""))
    if not key or not secret:
        raise ValueError("Alpaca credentials missing in Streamlit secrets.")
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    response = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{sym}/bars",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        params={"timeframe": tf, "start": (end - timedelta(days=days)).isoformat(),
                "end": end.isoformat(), "feed": "iex", "limit": 10000, "sort": "asc"},
        timeout=30,
    )
    response.raise_for_status()
    data = pd.DataFrame(response.json().get("bars", [])).rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    if data.empty:
        raise ValueError("Alpaca did not return bars for this selection.")
    data["timestamp"] = pd.to_datetime(data.timestamp, utc=True)
    return data.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

def ema(series, periods):
    return series.ewm(span=periods, adjust=False).mean()

def rsi(series, periods=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / periods, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / periods, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))

def technical_features(frame):
    x = frame.copy()
    for periods in [10, 20, 50, 100, 200]:
        x[f"ema{periods}"] = ema(x.close, periods)
    x["rsi"] = rsi(x.close)
    x["macd"] = ema(x.close, 12) - ema(x.close, 26)
    x["macds"] = ema(x.macd, 9)
    x["mom"] = x.close.pct_change(10)
    x["vr"] = x.volume / x.volume.rolling(20).mean()
    x["obv"] = (np.sign(x.close.diff()).fillna(0) * x.volume).cumsum()
    x["obvma"] = x.obv.rolling(20).mean()
    return x

def scanner_votes(x):
    latest, previous, rows = x.iloc[-1], x.iloc[-2], []
    for periods in [10, 20, 50, 100, 200]:
        vote = 1 if latest.close > latest[f"ema{periods}"] else -1
        rows.append([f"Price vs EMA{periods}", vote, f"{latest.close:.2f} vs {latest[f'ema{periods}']:.2f}"])
    vote = 1 if latest.rsi < 30 and latest.rsi > previous.rsi else -1 if latest.rsi > 70 and latest.rsi < previous.rsi else 0
    rows.append(["RSI(14)", vote, f"{latest.rsi:.1f}"])
    rows.append(["MACD", 1 if latest.macd > latest.macds else -1, f"{latest.macd:.3f} vs signal {latest.macds:.3f}"])
    rows.append(["Momentum 10", 1 if latest.mom > 0 else -1, f"{latest.mom * 100:.2f}%"])
    rows.append(["OBV trend", 1 if latest.obv > latest.obvma else -1, f"Relative volume {latest.vr:.2f}x"])
    body = abs(latest.close - latest.open)
    lower, upper = min(latest.open, latest.close) - latest.low, latest.high - max(latest.open, latest.close)
    hammer, shooting = lower > 2 * body and upper < body, upper > 2 * body and lower < body
    bull = latest.close > latest.open and previous.close < previous.open and latest.close >= previous.open and latest.open <= previous.close
    bear = latest.close < latest.open and previous.close > previous.open and latest.open >= previous.close and latest.close <= previous.open
    vote = 1 if hammer or bull else -1 if shooting or bear else 0
    rows.append(["Candlestick", vote, "Bullish reversal" if vote == 1 else "Bearish reversal" if vote == -1 else "No selected reversal"])
    return pd.DataFrame(rows, columns=["signal", "vote", "reason"])

def event_metrics(returns):
    values = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    gains, losses = values[values > 0].sum(), -values[values < 0].sum()
    return {"events": len(values), "expectancy_pct": 100 * values.mean() if len(values) else np.nan,
            "profit_factor": gains / losses if losses > 0 else np.nan,
            "win_rate_pct": 100 * (values > 0).mean() if len(values) else np.nan}

def evaluate_candidate(frame, family, fast, slow, window, threshold, horizon, split, cost_bps):
    x = frame.copy()
    trend = pd.Series(np.where(ema(x.close, fast) > ema(x.close, slow), 1.0, -1.0), index=x.index)
    if family == "EMA trend":
        signal = trend.where(trend.ne(trend.shift(1)), 0.0)
    else:
        previous_close = x.close.shift(1)
        tr = pd.concat([(x.high - x.low), (x.high - previous_close).abs(), (x.low - previous_close).abs()], axis=1).max(axis=1)
        compression = tr.rolling(window).mean() / tr.rolling(window * 4).mean()
        volume_z = (x.volume - x.volume.rolling(50).mean()) / x.volume.rolling(50).std(ddof=0).replace(0, np.nan)
        release = (compression.shift(1) < threshold) & (compression >= threshold) & (volume_z > 0)
        signal = trend.where(release, 0.0)
    future = x.close.shift(-horizon) / x.close - 1
    returns = (signal * future - signal.ne(0) * cost_bps / 10000).where(signal.ne(0))
    cut = int(len(x) * split)
    train, test = event_metrics(returns.iloc[:cut]), event_metrics(returns.iloc[cut:])
    return {"family": family, "fast": fast, "slow": slow, "compression_window": window,
            "threshold": threshold, "horizon_bars": horizon,
            **{f"train_{k}": v for k, v in train.items()}, **{f"test_{k}": v for k, v in test.items()}}

def edge_factory(frame, split, cost_bps):
    rows = []
    for fast, slow, horizon in product([8, 12, 16], [30, 40, 50], [1, 3, 5]):
        rows.append(evaluate_candidate(frame, "EMA trend", fast, slow, 12, 0.75, horizon, split, cost_bps))
    for fast, slow, window, threshold, horizon in product(
        [8, 12, 16], [30, 40, 50], [8, 12, 16], [0.65, 0.75, 0.85], [1, 3, 5]
    ):
        rows.append(evaluate_candidate(frame, "Compression expansion", fast, slow, window, threshold, horizon, split, cost_bps))
    results = pd.DataFrame(rows)
    results["plateau_score"] = np.nan
    for index, row in results.iterrows():
        nearby = results[(results.family == row.family) & (results.horizon_bars == row.horizon_bars)
                         & ((results.fast - row.fast).abs() <= 4) & ((results.slow - row.slow).abs() <= 10)
                         & ((results.threshold - row.threshold).abs() <= 0.11)].test_profit_factor.dropna()
        results.loc[index, "plateau_score"] = nearby.mean() if len(nearby) else np.nan
    results["candidate"] = ((results.train_events >= 30) & (results.test_events >= 20)
                            & (results.train_profit_factor > 1) & (results.test_profit_factor > 1)
                            & (results.plateau_score > 1))
    return results.sort_values(["candidate", "plateau_score", "test_profit_factor"], ascending=[False, False, False])

sym = st.sidebar.selectbox("Asset", ["SPY", "QQQ", "XLE", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "TSLA"])
tf = st.sidebar.selectbox("Timeframe", ["15Min", "1Hour", "1Day"], index=1)
days = st.sidebar.slider("History days", 30, 365, 120, 30)
scanner_tab, edge_tab, method_tab = st.tabs(["Technical Scanner", "Edge Factory", "Methodology"])

with scanner_tab:
    st.write("Baseline descrittiva: i voti non sono ancora pesi validati.")
    if st.button("Analyze technicals", key="scanner"):
        try:
            x = technical_features(getbars(sym, days, tf)); votes = scanner_votes(x)
            score = 100 * votes.vote.mean(); c1, c2, c3 = st.columns(3)
            c1.metric("BASELINE SCORE", f"{score:+.0f}"); c2.metric("Bullish", int((votes.vote == 1).sum())); c3.metric("Bearish", int((votes.vote == -1).sum()))
            st.line_chart(x.set_index("timestamp")[["close", "ema20", "ema50"]])
            votes["direction"] = votes.vote.map({1: "Bullish", 0: "Neutral", -1: "Bearish"})
            st.dataframe(votes[["signal", "direction", "reason"]], use_container_width=True, hide_index=True)
        except Exception as exc: st.error(str(exc))

with edge_tab:
    st.write("Ricerca condizionale con train/test cronologico, costi e stabilità dei parametri.")
    left, right = st.columns(2)
    split = left.slider("Train share", 0.55, 0.80, 0.70, 0.05)
    cost_bps = right.number_input("Cost per event (bps)", 0.0, 50.0, 2.0, 0.5)
    if st.button("Run Edge Factory", type="primary", key="edge"):
        try:
            bars = getbars(sym, days, tf)
            if len(bars) < 300: st.warning("Fewer than 300 bars: results are exploratory only.")
            with st.spinner("Testing candidate hypotheses…"): results = edge_factory(bars, split, cost_bps)
            st.metric("Preliminary candidates", int(results.candidate.sum()))
            st.dataframe(results, use_container_width=True, hide_index=True)
            st.download_button("Download results CSV", results.to_csv(index=False), "edge_factory_results.csv", "text/csv")
            st.warning("Event study, not a portfolio backtest. Candidates still require walk-forward and regime-conditioned Monte Carlo.")
        except Exception as exc: st.error(str(exc))

with method_tab:
    st.markdown("""**Isolation rule:** Technical Scanner is a baseline. Edge Factory does not inherit its score.

**Promotion rule:** train and unseen test must both be positive, contain enough events and survive nearby parameters.

**Not yet implemented:** futures/volume bars, regimes, walk-forward, Monte Carlo, news/events, portfolio simulation and paper execution.""")

st.caption("v0.7 • Technical Scanner + Edge Factory • No automatic orders")
