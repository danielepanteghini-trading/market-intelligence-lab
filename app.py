import os
from pathlib import Path
from datetime import timedelta
import numpy as np, pandas as pd, requests, streamlit as st
st.set_page_config(page_title="Market Intelligence Lab v0.3",layout="wide")
st.title("Market Intelligence Lab v0.3")
st.caption("Historical event testing + real Alpaca data. Research only.")
def bars(s,a,b):
 k=st.secrets.get("ALPACA_API_KEY",os.getenv("ALPACA_API_KEY","")); q=st.secrets.get("ALPACA_SECRET_KEY",os.getenv("ALPACA_SECRET_KEY",""))
 if not k or not q: raise RuntimeError("Alpaca keys not configured.")
 r=requests.get(f"https://data.alpaca.markets/v2/stocks/{s}/bars",headers={"APCA-API-KEY-ID":k,"APCA-API-SECRET-KEY":q},params={"timeframe":"1Min","start":a.isoformat(),"end":b.isoformat(),"feed":"iex","limit":10000,"sort":"asc"},timeout=30);r.raise_for_status()
 d=pd.DataFrame(r.json().get("bars",[])).rename(columns={"t":"timestamp","c":"close","v":"volume"})
 if d.empty:return d
 d["timestamp"]=pd.to_datetime(d.timestamp,utc=True);return d
def feat(d):
 x=d.copy();x["ret"]=x.close.pct_change();m=x.volume.rolling(60,min_periods=20);z=(x.volume-m.mean())/m.std();r=x.ret.rolling(60,min_periods=20);rz=(x.ret-r.mean())/r.std()
 x["anomaly_score"]=(100*(1-np.exp(-(.55*z.fillna(0).clip(lower=0)+.45*rz.abs().fillna(0))/3))).clip(0,100);return x
ev=pd.read_csv(Path(__file__).with_name("events_v03.csv"))
a,b,c=st.tabs(["Historical Test Engine","Event Catalog","Method"])
with a:
 u=ev[ev.status=="verified_seed"]; label=st.selectbox("Verified event",u.event_id+" — "+u.title); row=u[u.event_id==label.split(" — ")[0]].iloc[0]
 asset=st.selectbox("Asset",row.assets.split(";"));st.write("Event UTC:",row.event_time_utc)
 if st.button("Run historical test"):
  t=pd.Timestamp(row.event_time_utc)
  try:
   d=feat(bars(asset,t-timedelta(minutes=120),t+timedelta(minutes=120)))
   if d.empty: st.warning("No bars returned.")
   else:
    pre=d[d.timestamp<t];x,y,z=st.columns(3);x.metric("Pre max anomaly",f"{pre.anomaly_score.max():.1f}");y.metric("Pre alerts >75",int((pre.anomaly_score>=75).sum()));z.metric("Bars",len(d))
    st.line_chart(d.set_index("timestamp")[["close"]]);st.line_chart(d.set_index("timestamp")[["anomaly_score"]]);st.dataframe(d,use_container_width=True)
  except Exception as e:st.error(str(e))
with b: st.dataframe(ev,use_container_width=True);st.info("Events without an exact verified minute are excluded from minute tests.")
with c: st.markdown("**Protocol:** verify timestamps independently → test pre-event windows → matched controls → freeze rules → untouched holdout.")
st.caption("v0.3 • No automatic orders")
