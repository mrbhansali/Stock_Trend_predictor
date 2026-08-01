"""
Entry point. Run once a day from the terminal:

    python main.py

Each run:
  1. Evaluates the last pending saved prediction against what actually
     happened (logged to predictions.json for tracking accuracy over time)
  2. Fetches fresh data and gets the technical signal (XGBoost) + news
     signal (Gemini). The technical signal reports the last date its
     features are actually computed from (asof_date) — this is whatever
     the most recent complete trading session in the data is, which may
     or may not be "today" depending on when you run this.
  3. Fuses the two into one final BUY/SELL/HOLD call, dated for the NEXT
     NSE trading session after asof_date — this is a prediction FOR that
     day, made using only data available up to asof_date. It is never
     dated using datetime.now().
  4. Saves that fused prediction and prints a report. If a prediction for
     that same target date already exists (e.g. you ran this twice before
     new data landed), it's skipped instead of duplicated.

Note: the technical model does not retrain itself here. Run
`python technical_model.py` periodically as new data accumulates.
"""

import json
import os
from datetime import datetime, timedelta

import yfinance as yf
import exchange_calendars as xcals

import config
import technical_model
import news_signal

_nse_calendar = xcals.get_calendar("XBOM")


# ── PERSISTENCE ──────────────────────────────────────────────────────

def _load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def is_market_open(date_str: str) -> bool:
    return _nse_calendar.is_session(date_str)


def get_next_trading_day(asof_date) -> str:
    """First NSE trading session strictly AFTER asof_date — i.e. the date
    the prediction is actually for. asof_date is the last date the
    technical model's features are computed from (see technical_model.py),
    not today's system date."""
    asof_str = asof_date.strftime("%Y-%m-%d") if hasattr(asof_date, "strftime") else asof_date
    horizon_end = (datetime.strptime(asof_str, "%Y-%m-%d") + timedelta(days=14)).strftime("%Y-%m-%d")
    sessions = _nse_calendar.sessions_in_range(asof_str, horizon_end)
    for s in sessions:
        s_str = s.strftime("%Y-%m-%d")
        if s_str > asof_str:
            return s_str
    raise RuntimeError(f"Could not find an NSE trading session after {asof_str} within 14 days.")


def has_pending_prediction_for(pred_date: str) -> bool:
    predictions = _load_json(config.PREDICTIONS_FILE, default=[])
    return any(p["date"] == pred_date for p in predictions)


def get_last_pending_date():
    predictions = _load_json(config.PREDICTIONS_FILE, default=[])
    for pred in reversed(predictions):
        if not pred["evaluated"]:
            return pred["date"]
    return None


def _get_actual_change(ticker: str, pred_date: str):
    """% change from the close before pred_date to the first close on/after pred_date."""
    try:
        pred_dt = datetime.strptime(pred_date, "%Y-%m-%d")

        df_after = yf.download(
            ticker, start=pred_dt, end=pred_dt + timedelta(days=5),
            auto_adjust=True, progress=False,
        )
        if df_after.empty:
            return None
        first_close = df_after["Close"].iloc[0]
        if hasattr(first_close, "item"):
            first_close = first_close.item()

        df_before = yf.download(
            ticker, start=pred_dt - timedelta(days=10), end=pred_dt,
            auto_adjust=True, progress=False,
        )
        if df_before.empty:
            return None
        prev_close = df_before["Close"].iloc[-1]
        if hasattr(prev_close, "item"):
            prev_close = prev_close.item()

        return round(((first_close - prev_close) / prev_close) * 100, 3)

    except Exception as e:
        print(f"  Error fetching actual change for {ticker}: {e}")
        return None


def compute_reward(signal: str, actual_change: float, confidence: int):
    confidence_weight = (confidence or 0) / 100

    if signal == "BUY" and actual_change > 0:
        return round(1.0 * confidence_weight, 3), "CORRECT"
    if signal == "SELL" and actual_change < 0:
        return round(1.0 * confidence_weight, 3), "CORRECT"
    if signal == "HOLD":
        if abs(actual_change) < 1.0:
            return 0.3, "CORRECT"
        return -0.2, "MISSED MOVE"
    if signal == "BUY" and actual_change < 0:
        return round(-1.0 * confidence_weight, 3), "WRONG"
    if signal == "SELL" and actual_change > 0:
        return round(-1.0 * confidence_weight, 3), "WRONG"
    return 0.0, "NEUTRAL"


def evaluate_pending_prediction():
    pred_date = get_last_pending_date()
    if not pred_date:
        print("  No pending prediction to evaluate.")
        return

    predictions = _load_json(config.PREDICTIONS_FILE, default=[])

    for pred in predictions:
        if pred["evaluated"] or pred["date"] != pred_date:
            continue

        actual_change = _get_actual_change(pred["ticker"], pred_date)
        if actual_change is None:
            print(f"  Skipping {pred['ticker']} — market data not available yet.")
            continue

        reward, outcome = compute_reward(pred["signal"], actual_change, pred["confidence"])
        direction = "rose" if actual_change > 0 else "fell"

        pred["evaluated"] = True
        pred["outcome"] = outcome
        pred["reward"] = reward
        pred["actual_change"] = actual_change
        print(f"  {pred['ticker']}: {outcome} (reward {reward:+.2f}) — stock {direction} {abs(actual_change):.2f}%")

    _save_json(config.PREDICTIONS_FILE, predictions)


def save_prediction(fused: dict, pred_date: str):
    existing = _load_json(config.PREDICTIONS_FILE, default=[])
    existing.append({
        "ticker": fused["ticker"],
        "signal": fused["signal"],
        "confidence": fused["confidence"],
        "date": pred_date,
        "evaluated": False,
    })
    _save_json(config.PREDICTIONS_FILE, existing)


# ── FUSION ────────────────────────────────────────────────────────────

def _signal_to_score(signal: str, confidence) -> float:
    direction = {"BUY": 1, "SELL": -1, "HOLD": 0}.get(signal, 0)
    return direction * ((confidence or 0) / 100)


def fuse_signals(tech: dict, news: dict) -> dict:
    tech_score = _signal_to_score(tech["signal"], tech["confidence"])
    news_score = _signal_to_score(news["signal"], news["confidence"])

    combined = config.TECHNICAL_WEIGHT * tech_score + config.NEWS_WEIGHT * news_score

    if combined > config.BUY_THRESHOLD:
        final_signal = "BUY"
    elif combined < config.SELL_THRESHOLD:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    final_confidence = min(round(abs(combined) * 100), 100)

    conflict = (
        abs(tech_score) >= config.CONFLICT_MAGNITUDE
        and abs(news_score) >= config.CONFLICT_MAGNITUDE
        and (tech_score > 0) != (news_score > 0)
    )

    if conflict:
        final_confidence = min(final_confidence, config.CONFLICT_CONFIDENCE_CAP)
        risk = "HIGH"
    else:
        risk = tech.get("risk") or "MEDIUM"

    return {
        "ticker": config.TICKER,
        "signal": final_signal,
        "confidence": final_confidence,
        "risk": risk,
        "conflict": conflict,
        "combined_score": round(combined, 3),
        "technical": tech,
        "news": news,
    }


# ── REPORT ────────────────────────────────────────────────────────────

def print_report(fused: dict, pred_date: str):
    print("\n" + "=" * 60)
    print(f"  FINAL SIGNAL — {fused['ticker']}   {pred_date}")
    print("=" * 60)
    print(f"  Signal      : {fused['signal']}")
    print(f"  Confidence  : {fused['confidence']}%")
    print(f"  Risk        : {fused['risk']}")
    if fused["conflict"]:
        print(f"  ⚠ Technical and news signals disagree — confidence capped.")
    print(f"  Combined score (weighted, -1 to +1): {fused['combined_score']:+.3f}")

    print("\n  -- Technical signal --")
    print(f"  {fused['technical']['signal']} @ {fused['technical']['confidence']}% "
          f"(risk: {fused['technical']['risk']})")
    for point in fused["technical"].get("reasoning", []):
        print(f"    - {point}")

    print("\n  -- News/macro signal --")
    print(f"  {fused['news']['signal']} @ {fused['news']['confidence']}%")
    for point in fused["news"].get("reasoning", []):
        print(f"    - {point}")

    print("=" * 60)


# ── RUN ───────────────────────────────────────────────────────────────

def run_today():
    print("[1/3] Evaluating pending prediction (if any)...")
    evaluate_pending_prediction()

    print("\n[2/3] Fetching signals...")
    tech = technical_model.predict_today(config.TICKER)
    news = news_signal.get_news_signal()

    # The prediction is FOR the next trading session after the data we
    # actually have (tech["asof_date"]) — not for whatever today's
    # calendar date happens to be.
    pred_date = get_next_trading_day(tech["asof_date"])

    if has_pending_prediction_for(pred_date):
        print(f"\nAlready have a prediction for {pred_date} (latest data is still as of "
              f"{tech['asof_date'].date()}) — nothing new to do until fresh data is available.")
        return

    print("\n[3/3] Fusing signals...")
    fused = fuse_signals(tech, news)
    save_prediction(fused, pred_date)

    print_report(fused, pred_date)
    print(f"\nDone. This predicts {pred_date} (based on data as of {tech['asof_date'].date()}). "
          f"Run again after {pred_date} to evaluate it.")


if __name__ == "__main__":
    run_today()