import os
from datetime import datetime,timedelta,timezone
import numpy as np,pandas as pd,requests,streamlit as st
st.set_page_config(page_title="Market Intelligence Lab v0.6",layout="wide")
st.title("Market Intelligence Lab v0.6 — Technical Engine")
st.caption("Explainable technical consensus • research only")
def getbars(sym,days,tf):
 k=st.secrets.get("ALPACA_API_KEY",os.getenv("ALPACA_API_KEY",""));s=st.secrets.get("ALPACA_SECRET_KEY",os.getenv("ALPACA_SECRET_KEY",""))
 end=datetime.now(timezone.utc)-timedelta(minutes=16);start=end-timedelta(days=days)
 r=requests.get(f"https://data.alpaca.markets/v2/stocks/{sym}/bars",headers={"APCA-API-KEY-ID":k,"APCA-API-SECRET-KEY":s},params={"timeframe":tf,"start":start.isoformat(),"end":end.isoformat(),"feed":"iex","limit":10000,"sort":"asc"},timeout=30);r.raise_for_status()
 d=pd.DataFrame(r.json().get("bars",[])).rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"});d["timestamp"]=pd.to_datetime(d.timestamp,utc=True);return d
def ema(s,n):return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
 d=s.diff();u=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean();q=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean();return 100-100/(1+u/q.replace(0,np.nan))
def features(x):
 c=x.close
 for n in [10,20,50,100,200]:x[f"ema{n}"]=ema(c,n)
 x["rsi"]=rsi(c);x["macd"]=ema(c,12)-ema(c,26);x["macds"]=ema(x.macd,9);x["mom"]=c.pct_change(10)
 x["vr"]=x.volume/x.volume.rolling(20).mean()
 sign=np.sign(c.diff()).fillna(0);x["obv"]=(sign*x.volume).cumsum();x["obvma"]=x.obv.rolling(20).mean()
 return x
sym=st.sidebar.selectbox("Asset",["SPY","QQQ","XLE","GLD","TLT","AAPL","MSFT","NVDA","TSLA"])
tf=st.sidebar.selectbox("Timeframe",["15Min","1Hour","1Day"],index=1);days=st.sidebar.slider("History days",30,365,120,30)
if st.button("Analyze technicals"):
 try:
  x=features(getbars(sym,days,tf));z=x.iloc[-1];p=x.iloc[-2];rows=[]
  for n in [10,20,50,100,200]:
   q=1 if z.close>z[f"ema{n}"] else -1;rows.append([f"Price vs EMA{n}",q,f"{z.close:.2f} vs {z[f'ema{n}']:.2f}"])
  q=1 if z.rsi<30 and z.rsi>p.rsi else -1 if z.rsi>70 and z.rsi<p.rsi else 0;rows.append(["RSI(14)",q,f"{z.rsi:.1f}"])
  q=1 if z.macd>z.macds else -1;rows.append(["MACD",q,f"{z.macd:.3f} vs signal {z.macds:.3f}"])
  q=1 if z.mom>0 else -1;rows.append(["Momentum 10",q,f"{z.mom*100:.2f}%"])
  q=1 if z.obv>z.obvma else -1;rows.append(["OBV trend",q,f"Relative volume {z.vr:.2f}x"])
  body=abs(z.close-z.open);lo=min(z.open,z.close)-z.low;up=z.high-max(z.open,z.close)
  hammer=lo>2*body and up<body;shoot=up>2*body and lo<body
  engb=z.close>z.open and p.close<p.open and z.close>=p.open and z.open<=p.close
  engs=z.close<z.open and p.close>p.open and z.open>=p.close and z.close<=p.open
  q=1 if hammer or engb else -1 if shoot or engs else 0;rows.append(["Candlestick",q,"Bullish reversal" if q==1 else "Bearish reversal" if q==-1 else "No selected reversal"])
  v=pd.DataFrame(rows,columns=["signal","vote","reason"]);score=100*v.vote.mean()
  a,b,c=st.columns(3);a.metric("TECHNICAL SCORE",f"{score:+.0f}");b.metric("Bullish",(v.vote==1).sum());c.metric("Bearish",(v.vote==-1).sum())
  st.line_chart(x.set_index("timestamp")[["close","ema20","ema50"]])
  v["direction"]=v.vote.map({1:"Bullish",0:"Neutral",-1:"Bearish"});st.dataframe(v[["signal","direction","reason"]],use_container_width=True)
  st.info("Baseline only: next step is backtesting each signal and combinations against future returns.")
 except Exception as e:st.error(str(e))
st.caption("v0.6 • Technical Engine • No automatic orders")
