import os
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Market Intelligence Lab v0.4", layout="wide")
st.title("Market Intelligence Lab v0.4")
st.caption("Event-vs-control historical testing with real Alpaca data. Research only.")

THRESHOLD = 75

def alpaca_bars(symbol, start, end):
    key = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY",""))
    secret = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY",""))
    if not key or not secret:
        raise RuntimeError("Alpaca keys not configured.")
    url=f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    headers={"APCA-API-KEY-ID":key,"APCA-API-SECRET-KEY":secret}
    params={"timeframe":"1Min","start":start.isoformat(),"end":end.isoformat(),
            "feed":"iex","limit":10000,"sort":"asc"}
    rows=[]
    while True:
        r=requests.get(url,headers=headers,params=params,timeout=30)
        r.raise_for_status()
        j=r.json(); rows += j.get("bars",[])
        token=j.get("next_page_token")
        if not token: break
        params["page_token"]=token
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows).rename(columns={"t":"timestamp","o":"open","h":"high","l":"low",
                                         "c":"close","v":"volume","vw":"vwap","n":"trades"})
    d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
    return d

def features(d):
    x=d.sort_values("timestamp").copy()
    x["ret"]=x["close"].pct_change()
    vm=x["volume"].rolling(60,min_periods=20)
    rm=x["ret"].rolling(60,min_periods=20)
    x["volume_z"]=(x["volume"]-vm.mean())/vm.std()
    x["return_z"]=(x["ret"]-rm.mean())/rm.std()
    rv=x["ret"].rolling(15,min_periods=5).std()
    rvm=rv.rolling(60,min_periods=20)
    x["volatility_z"]=(rv-rvm.mean())/rvm.std()
    raw=.45*x["volume_z"].fillna(0).clip(lower=0)+.35*x["return_z"].abs().fillna(0)+.20*x["volatility_z"].fillna(0).clip(lower=0)
    x["anomaly_score"]=(100*(1-np.exp(-raw/3))).clip(0,100)
    return x

def window_stats(d, t, minutes=60):
    p=d[(d.timestamp<t)&(d.timestamp>=t-pd.Timedelta(minutes=minutes))]
    if p.empty: return None
    return {"max":float(p.anomaly_score.max()),"mean":float(p.anomaly_score.mean()),
            "alerts":int((p.anomaly_score>=THRESHOLD).sum()),"bars":len(p)}

def matched_control_times(t, n=12):
    # Same weekday and UTC clock time, surrounding weeks, excluding event day.
    vals=[]
    for w in range(1,9):
        vals += [t-pd.Timedelta(days=7*w), t+pd.Timedelta(days=7*w)]
    return vals[:n]

events=pd.read_csv(Path(__file__).with_name("events_v04.csv"))
tabs=st.tabs(["Single Event","Matched Controls","Batch Overview","Event Catalog","Method"])

with tabs[0]:
    u=events[events.status=="verified"]
    label=st.selectbox("Event",u.event_id+" — "+u.title,key="single_event")
    row=u[u.event_id==label.split(" — ")[0]].iloc[0]
    asset=st.selectbox("Asset",row.assets.split(";"),key="single_asset")
    t=pd.Timestamp(row.event_time_utc)
    st.write("Event UTC:",row.event_time_utc," | Session:",row.session)
    if st.button("Run event window"):
        try:
            d=features(alpaca_bars(asset,t-timedelta(minutes=150),t+timedelta(minutes=90)))
            s=window_stats(d,t,60)
            if not s: st.warning("Insufficient bars.")
            else:
                a,b,c=st.columns(3);a.metric("T-60 max",f"{s['max']:.1f}");b.metric("T-60 mean",f"{s['mean']:.1f}");c.metric("Alerts >75",s["alerts"])
                st.line_chart(d.set_index("timestamp")[["close"]])
                st.line_chart(d.set_index("timestamp")[["anomaly_score"]])
                st.dataframe(d,use_container_width=True)
        except Exception as e: st.error(str(e))

with tabs[1]:
    st.subheader("Event vs matched normal windows")
    label=st.selectbox("Event",u.event_id+" — "+u.title,key="control_event")
    row=u[u.event_id==label.split(" — ")[0]].iloc[0]
    asset=st.selectbox("Asset",row.assets.split(";"),key="control_asset")
    n=st.slider("Number of matched controls",4,16,12,2)
    if st.button("Run matched-control test"):
        t=pd.Timestamp(row.event_time_utc)
        try:
            ed=features(alpaca_bars(asset,t-timedelta(minutes=150),t+timedelta(minutes=5)))
            es=window_stats(ed,t,60)
            results=[]
            for ct in matched_control_times(t,n):
                try:
                    cd=features(alpaca_bars(asset,ct-timedelta(minutes=150),ct+timedelta(minutes=5)))
                    cs=window_stats(cd,ct,60)
                    if cs: results.append({"control_time":ct,"max_score":cs["max"],"mean_score":cs["mean"],"alerts":cs["alerts"]})
                except: pass
            rdf=pd.DataFrame(results)
            if es and not rdf.empty:
                percentile=100*(rdf.max_score < es["max"]).mean()
                lift=(es["max"]/(rdf.max_score.median()+1e-9))
                a,b,c,d=st.columns(4)
                a.metric("Event max",f"{es['max']:.1f}")
                b.metric("Control median max",f"{rdf.max_score.median():.1f}")
                c.metric("Event percentile",f"{percentile:.0f}%")
                d.metric("Max-score ratio",f"{lift:.2f}x")
                st.bar_chart(pd.DataFrame({"max_score":[es["max"]]+rdf.max_score.tolist()},
                    index=["EVENT"]+[f"C{i+1}" for i in range(len(rdf))]))
                st.dataframe(rdf,use_container_width=True)
            else: st.warning("Not enough comparable bars were returned.")
        except Exception as e: st.error(str(e))

with tabs[2]:
    st.subheader("Batch Overview")
    st.write("Next milestone: run all verified events × assets and calculate aggregate predictive lift.")
    st.info("We do not call a pattern predictive until it beats matched controls across many independent events.")

with tabs[3]:
    st.dataframe(events,use_container_width=True)

with tabs[4]:
    st.markdown("""
**v0.4 protocol**
1. Timestamp the public event independently from market movement.
2. Measure T−60 anomaly behavior.
3. Compare against same-asset, same-weekday, same-clock-time control windows.
4. Separate market-hours and after-hours events.
5. Later add volatility-regime matching, cross-market confirmation and an untouched holdout set.

A high score in one famous event is not evidence of prediction by itself.
""")
st.caption("v0.4 • No automatic orders • No investment recommendation")
