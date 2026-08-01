# Fused Technical + News Trading Signal (NSE)

Daily BUY / SELL / HOLD signal for a single NSE-listed stock, produced by
fusing two independent models:

- **Technical signal** — an XGBoost classifier trained on price/volume
  indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, volume
  ratio), predicting whether the next trading day's close will be higher
  or lower than the latest available close.
- **News/macro signal** — recent headlines (via NewsAPI) filtered for
  relevance and scored for sentiment by Gemini, combined with sector
  macro data (a comparable index, USD/INR, a competitor stock), turned
  into a BUY/SELL/HOLD call by an LLM prompt.

The two are combined into one final call with a confidence score and a
risk rating, logged to `predictions.json`, and automatically graded
against what actually happened once enough time has passed.

> ⚠️ **Disclaimer:** This is a research/educational project, not
> financial advice. Nothing here should be the sole basis for an
> investment decision. Past performance in a backtest does not predict
> future results, and this system has real, documented limitations
> (see [Limitations](#limitations) below).

---

## Repo structure

| File | Purpose |
|---|---|
| `feature_extraction.py` | Fetches OHLCV data (yfinance) and computes technical indicators. No LLM involved. |
| `technical_model.py` | Trains and serves the XGBoost classifier. Run standalone to (re)train. |
| `news_signal.py` | Fetches news + macro data, scores sentiment, asks Gemini for a signal. |
| `main.py` | Daily entry point — evaluates yesterday's call, fetches both signals, fuses them, saves + prints a report. |
| `config.py` | **You create this** (not committed — see below). Ticker, API keys, weights, thresholds. |
| `config.py.example` | Template for the above. |
| `models/` | Auto-created; holds the saved trained XGBoost model. |

---

## Setup

**1. Requirements**

```bash
pip install -r requirements.txt
```

Needs Python 3.10+.

**2. Configuration**

```bash
cp config.py.example config.py
```

Then edit `config.py` with your ticker, company/sector names, macro
comparables, and API keys (Gemini + [NewsAPI.org](https://newsapi.org)).
Every variable it needs is documented inline in `config.py.example`.

`config.py` is listed in `.gitignore` — **do not commit it**, since it
holds your API keys. If you fork or share this repo, only
`config.py.example` should be public.

**3. Train the technical model (once, before first use)**

```bash
python technical_model.py
```

This fetches history, does a **chronological** train/test split (not
shuffled — shuffling time-series data would leak future information into
training), trains the XGBoost model on the earlier portion, reports
holdout accuracy on the later portion, and saves the model to
`models/xgb_<ticker>.json`.

Re-run this periodically as new trading days accumulate — the model does
not retrain itself automatically.

---

## Running it day-to-day

```bash
python main.py
```

Each run:
1. Evaluates the last pending prediction in `predictions.json` (if the
   outcome is now known).
2. Fetches fresh data and gets both signals.
3. Fuses them into one BUY/SELL/HOLD call.
4. Saves and prints the result.

**Important:** the prediction is always dated for the **next trading
session after the most recent data actually available** — never for
whatever the system clock says "today" is. This means you can run it any
day (including weekends) and it will correctly report what it's
predicting for, and it won't write a duplicate entry if you run it twice
before new data lands.

---

## Limitations

- Single ticker, relatively small dataset (a few years of daily data) —
  XGBoost can overfit fast; `technical_model.py` prints a warning if it
  isn't meaningfully beating a majority-class baseline.
- No automatic retraining — you decide when to re-run
  `python technical_model.py`.
- The news signal depends on NewsAPI's free tier only covering roughly
  the last month of articles, so its usefulness is limited to live,
  current-day use rather than any kind of historical replay.
- Overall fused-pipeline performance is only ever tracked going forward,
  via `predictions.json` — there's no way to validate it against history.

---
