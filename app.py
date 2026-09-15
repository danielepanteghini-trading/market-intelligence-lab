import os
from pathlib import Path
from datetime import timedelta
import numpy as np,pandas as pd,requests,streamlit as st

st.set_page_config(page_title="Market Intelligence Lab v0.5",layout="wide")
st.title("Market Intelligence Lab v0.5")
st.caption("Robust historical event-vs-control testing • real Alpaca data • research only")
TH=75; WINDOWS=[60,30,15,5,1]; WARMUP_MIN=420

def bars(symbol,start,end):
 k=st.secrets.get("ALPACA_API_KEY",os.getenv("ALPACA_API_KEY","")); s=st.secrets.get("ALPACA_SECRET_KEY",os.getenv("ALPACA_SECRET_KEY",""))
 if not k or not s: raise RuntimeError("Alpaca keys not configured.")
 p={"timeframe":"1Min","start":start.isoformat(),"end":end.isoformat(),"feed":"iex","limit":10000,"sort":"asc"}; rows=[]
 while True:
  r=requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",headers={"APCA-API-KEY-ID":k,"APCA-API-SECRET-KEY":s},params=p,timeout=30); r.raise_for_status()
  j=r.json(); rows+=j.get("bars",[]); token=j.get("next_page_token")
  if not token: break
  p["page_token"]=token
 if not rows:return pd.DataFrame()
 d=pd.DataFrame(rows).rename(columns={"t":"timestamp","c":"close","v":"volume"})
 d["timestamp"]=pd.to_datetime(d.timestamp,utc=True); return d

def feat(d):
 x=d.sort_values("timestamp").copy(); x["ret"]=x.close.pct_change()
 vm=x.volume.rolling(120,min_periods=60); rm=x.ret.rolling(120,min_periods=60)
 x["vz"]=(x.volume-vm.mean())/vm.std(); x["rz"]=(x.ret-rm.mean())/rm.std()
 rv=x.ret.rolling(20,min_periods=10).std(); rvm=rv.rolling(120,min_periods=60)
 x["volz"]=(rv-rvm.mean())/rvm.std()
 raw=.45*x.vz.fillna(0).clip(lower=0)+.35*x.rz.abs().fillna(0)+.20*x.volz.fillna(0).clip(lower=0)
 x["score"]=(100*(1-np.exp(-raw/3))).clip(0,100); return x

def get_window(symbol,t):
 # Several calendar days guarantee enough market-session warm-up even across weekends.
 d=feat(bars(symbol,t-pd.Timedelta(days=4),t+pd.Timedelta(minutes=10)))
 return d

def stats(d,t,m):
 p=d[(d.timestamp<t)&(d.timestamp>=t-pd.Timedelta(minutes=m))].copy()
 # Need meaningful coverage and initialized rolling features.
 if p.empty or len(p)<max(1,int(m*.55)): return None
 valid=p[["vz","rz","volz"]].notna().any(axis=1).sum()
 if valid<max(1,int(len(p)*.5)): return None
 return {"max":float(p.score.max()),"mean":float(p.score.mean()),"alerts":int((p.score>=TH).sum()),"bars":len(p)}

def candidate_controls(t):
 vals=[]
 for w in range(1,15):
  vals.extend([t-pd.Timedelta(days=7*w),t+pd.Timedelta(days=7*w)])
 return vals

ev=pd.read_csv(Path(__file__).with_name("events_v05.csv")); u=ev[ev.status=="verified"]
tabs=st.tabs(["Event Windows","Matched Controls","Event Catalog","Method"])

with tabs[0]:
 lab=st.selectbox("Event",u.event_id+" — "+u.title,key="e1"); row=u[u.event_id==lab.split(" — ")[0]].iloc[0]
 asset=st.selectbox("Asset",row.assets.split(";"),key="a1"); t=pd.Timestamp(row.event_time_utc)
 st.write("Testing:",row.event_id,"|",row.event_time_utc,"|",row.session)
 if st.button("Run robust event test"):
  try:
   d=get_window(asset,t); cols=st.columns(5)
   for i,m in enumerate(WINDOWS):
    q=stats(d,t,m); cols[i].metric(f"T-{m} max","N/A" if not q else f"{q['max']:.1f}")
   plot=d[(d.timestamp>=t-pd.Timedelta(minutes=90))&(d.timestamp<=t+pd.Timedelta(minutes=10))]
   st.line_chart(plot.set_index("timestamp")[["close"]]); st.line_chart(plot.set_index("timestamp")[["score"]])
  except Exception as e:st.error(str(e))

with tabs[1]:
 lab=st.selectbox("Event",u.event_id+" — "+u.title,key="e2"); row=u[u.event_id==lab.split(" — ")[0]].iloc[0]
 asset=st.selectbox("Asset",row.assets.split(";"),key="a2"); requested=st.slider("Valid matched controls required",4,12,8)
 st.write("Selected:",row.event_id,"|",row.event_time_utc,"|",row.session)
 if st.button("Run validated controls"):
  t=pd.Timestamp(row.event_time_utc)
  try:
   ed=get_window(asset,t); es={m:stats(ed,t,m) for m in WINDOWS}
   controls=[]; rejected=0
   for ct in candidate_controls(t):
    if len(controls)>=requested:break
    try:
     cd=get_window(asset,ct); cs={m:stats(cd,ct,m) for m in WINDOWS}
     if all(cs[m] is not None for m in WINDOWS):
      controls.append({"time":ct,**{f"max_{m}":cs[m]["max"] for m in WINDOWS}})
     else: rejected+=1
    except: rejected+=1
   rdf=pd.DataFrame(controls)
   st.metric("Valid controls",f"{len(rdf)}/{requested}"); st.caption(f"Rejected invalid candidates: {rejected}")
   if len(rdf)<requested or any(es[m] is None for m in WINDOWS):
    st.error("Inference blocked: insufficient valid event/control windows.")
   else:
    rows=[]
    for m in WINDOWS:
     em=es[m]["max"]; med=float(rdf[f"max_{m}"].median()); pct=100*float((rdf[f"max_{m}"]<em).mean())
     ratio=np.nan if med<=0 else em/med
     rows.append({"window":f"T-{m}","event_max":em,"control_median":med,"percentile":pct,"ratio":ratio})
    res=pd.DataFrame(rows)
    st.dataframe(res,use_container_width=True)
    st.bar_chart(res.set_index("window")[["event_max","control_median"]])
    st.dataframe(rdf,use_container_width=True)
  except Exception as e:st.error(str(e))

with tabs[2]: st.dataframe(ev,use_container_width=True)
with tabs[3]:
 st.markdown("""**v0.5 safeguards:** multi-day warm-up; rolling features require real history; invalid controls are rejected; inference is blocked unless the requested number of controls is valid; zero-denominator ratios are never reported; T−60/T−30/T−15/T−5/T−1 are evaluated separately.

**Important:** same-weekday/same-clock-time matching is only an intermediate control method. Before any claim of predictive power we still need volatility-regime matching, multiple-testing correction, cross-market features and an untouched out-of-sample holdout.""")
st.caption("v0.5 • No automatic orders • No investment recommendation")
