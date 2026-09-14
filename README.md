
# Anomaly Lab v0.1

A research prototype for testing whether statistically unusual market activity
appears disproportionately often before major political, geopolitical, macro or
corporate announcements.

## What it does
- Demo replay with 1-minute bars
- Optional Alpaca historical IEX bars
- Transparent anomaly score using volume, return and volatility z-scores
- Seed event catalog
- Research plan for event-vs-control testing

## What it does NOT do
- No live orders
- No investment recommendations
- No claim that an anomaly implies insider information
- No use of unverified event timestamps in final statistics

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Alpaca keys
Create `.streamlit/secrets.toml`:

```toml
ALPACA_API_KEY = "..."
ALPACA_SECRET_KEY = "..."
```

The app uses `feed=iex` for the free plan.

## Deploy free
Push these files to a GitHub repository and deploy it with Streamlit Community Cloud.

## Next milestones
1. Verify exact timestamps for 50–100 events.
2. Download event windows for SPY, QQQ, XLE, GLD, TLT.
3. Add matched random control windows.
4. Measure false-positive rates and predictive lift.
5. Add cross-asset anomaly score.
6. Add news/public-source ingestion only after the quantitative baseline works.
