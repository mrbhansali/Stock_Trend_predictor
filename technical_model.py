"""
Technical signal: trains and serves an XGBoost classifier on the
features from feature_extraction.py to predict next-day direction.

This replaces the earlier LLM-on-raw-prices approach.

Train (do this first, and re-run periodically as new data accumulates):
    python technical_model.py

main.py then calls predict_today() daily using the saved model —
it does NOT retrain automatically, so you know exactly when the
model's view of the world last updated.
"""

import os

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

import config
import feature_extraction as fx

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, f"xgb_{config.TICKER.replace('.', '_')}.json")

# Probability band around 0.5 that we treat as "no real edge" -> HOLD,
# rather than forcing a low-conviction BUY/SELL out of a coin-flip.
HOLD_DEADBAND = 0.05


def train_model(ticker: str = config.TICKER, period: str = "5y", test_frac: float = 0.2):
    X, y, dates = fx.build_dataset(ticker, period)

    # Chronological split — NOT shuffled. Shuffling time-series data leaks
    # future information into training and makes the accuracy number
    # meaningless. This is the single most important line in this file.
    split_idx = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training on {len(X_train)} rows ({dates[0].date()} to {dates[split_idx - 1].date()})")
    print(f"Testing on  {len(X_test)} rows ({dates[split_idx].date()} to {dates[-1].date()})")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,          # kept shallow deliberately — single-ticker,
        learning_rate=0.05,    # ~1000-row dataset overfits fast with more depth
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    baseline = max(y_test.mean(), 1 - y_test.mean())  # always-predict-majority-class

    print("\n" + "=" * 55)
    print("  HOLDOUT EVALUATION (chronological, out-of-sample)")
    print("=" * 55)
    print(f"  Accuracy           : {acc:.3f}")
    print(f"  Precision (up)     : {prec:.3f}")
    print(f"  Recall (up)        : {rec:.3f}")
    print(f"  Majority baseline  : {baseline:.3f}   <- always predicting the more common class")
    print(f"  Confusion matrix   :\n{confusion_matrix(y_test, preds)}")
    if acc <= baseline + 0.02:
        print("\n  NOTE: model is not meaningfully beating the majority-class baseline.")
        print("  That means these features aren't carrying signal above the base")
        print("  rate on this stock/period — a real result, not a bug to fix by")
        print("  tuning harder. Worth knowing before you trust this in the fused signal.")
    print("=" * 55)

    importances = sorted(
        zip(fx.FEATURE_COLUMNS, model.feature_importances_),
        key=lambda item: item[1], reverse=True,
    )
    print("\n  Top features by importance:")
    for name, score in importances[:5]:
        print(f"    {name:<16} {score:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": model, "importances": importances}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


def _load():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python technical_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def predict_today(ticker: str = config.TICKER) -> dict:
    saved = _load()
    model = saved["model"]
    importances = saved["importances"]

    features, asof_date = fx.get_latest_feature_row(ticker)
    prob_up = float(model.predict_proba(features)[0][1])

    if prob_up > 0.5 + HOLD_DEADBAND:
        signal = "BUY"
        confidence = round((prob_up - 0.5) * 2 * 100)
    elif prob_up < 0.5 - HOLD_DEADBAND:
        signal = "SELL"
        confidence = round((0.5 - prob_up) * 2 * 100)
    else:
        signal = "HOLD"
        confidence = round((HOLD_DEADBAND - abs(prob_up - 0.5)) / HOLD_DEADBAND * 100)

    if confidence >= 60:
        risk = "LOW"
    elif confidence >= 30:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    top_features = ", ".join(name for name, _ in importances[:3])

    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "risk": risk,
        "asof_date": asof_date,  # last date the input features are computed from
        "reasoning": [
            f"P(up) = {prob_up:.3f} from XGBoost classifier "
            f"(features as of {asof_date.date()}, predicting the next trading day)",
            f"Model's globally most-weighted features: {top_features}",
            "Note: this is global feature importance, not a per-day causal explanation.",
        ],
        "raw": {"prob_up": prob_up},
    }


if __name__ == "__main__":
    train_model()