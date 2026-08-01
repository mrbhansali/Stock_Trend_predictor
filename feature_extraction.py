"""
Feature extraction for the technical signal: fetches OHLCV data and
computes technical indicators using the `ta` library.

No LLM involved here — this produces a numeric feature matrix for a
classical ML model (see technical_model.py).
"""

import pandas as pd
import yfinance as yf
import ta

FEATURE_COLUMNS = [
    "return_1d",
    "sma_5", "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_signal", "macd_diff",
    "bb_width", "bb_pct",
    "atr_14",
    "vol_ratio_20d",
    "obv_change",
]


def fetch_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1d"] = df["Close"].pct_change() * 100

    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()

    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"] = bb.bollinger_pband()

    df["atr_14"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14
    ).average_true_range()

    df["vol_ratio_20d"] = df["Volume"] / df["Volume"].rolling(20).mean()

    obv = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["obv_change"] = obv.pct_change() * 100

    return df


def build_dataset(ticker: str, period: str = "5y"):
    """Returns (X, y, dates) for training.

    y = 1 if next day's close is higher than today's close, else 0.
    (Today's feature row predicts tomorrow's direction — that's the
    whole point of shift(-1) here.)
    """
    df = fetch_ohlcv(ticker, period)
    df = add_indicators(df)

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target"])

    X = df[FEATURE_COLUMNS]
    y = df["target"]
    dates = df.index

    return X, y, dates


def get_latest_feature_row(ticker: str, period: str = "1y"):
    """Single-row DataFrame of the most recent feature values, for a live
    prediction, PLUS the date that row's data is as-of.

    Important: this is NOT "today". It's the most recent COMPLETE trading
    day present in the fetched data (yfinance won't have a real close for
    a session that hasn't finished yet). Because the model is trained to
    predict tomorrow-vs-today (see build_dataset's shift(-1) target), a
    prediction made from this row is always a prediction for the trading
    day AFTER `asof_date` — never for asof_date itself, and never for
    whatever the system clock says "today" is.

    Returns:
        (feature_row_df, asof_date) where asof_date is a pandas Timestamp.
    """
    df = fetch_ohlcv(ticker, period)
    df = add_indicators(df)
    df = df.dropna(subset=FEATURE_COLUMNS)

    if df.empty:
        raise ValueError(
            f"Not enough history for {ticker} to compute all indicators. "
            f"Try a longer period."
        )

    asof_date = df.index[-1]
    return df[FEATURE_COLUMNS].iloc[[-1]], asof_date