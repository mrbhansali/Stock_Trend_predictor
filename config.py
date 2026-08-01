"""
Central configuration: API keys (from .env), ticker/sector settings,
signal-fusion parameters, and file paths.

Nothing in this file is secret. Real keys live in a local .env file
(see .env.example) which is NOT committed to git.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current working directory

# ── API KEYS ─────────────────────────────────────────────────────────
# Note: no Groq key here — the technical signal is now XGBoost, not an
# LLM call, so only the news/macro pipeline needs API keys.
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_REQUIRED_KEYS = {
    "NEWS_API_KEY": NEWS_API_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}
_missing = [k for k, v in _REQUIRED_KEYS.items() if not v]
if _missing:
    raise RuntimeError(
        f"Missing required API key(s) in .env: {', '.join(_missing)}. "
        f"Copy .env.example to .env and fill in real values."
    )

# ── TICKER / SECTOR (single ticker for now) ─────────────────────────
TICKER = "TCS.NS"          # yfinance ticker
COMPANY_NAME = "TCS"       # used in prompts / news search
SECTOR = "Indian IT"

SECTOR_CONTEXT = """
These are Indian IT sector stocks. Key macro triggers that affect this sector:
- USD/INR exchange rate (revenue is USD, costs are INR)
- US economy health (their biggest client market)
- Quarterly earnings and deal wins
- US Fed rate decisions (affect client IT budgets)
- Attrition rates and hiring trends
"""

MACRO_TICKERS = {
    "NIFTY 50": "^NSEI",
    "NIFTY IT Index": "^CNXIT",
    "USD/INR": "INR=X",
    "EUR/INR": "EURINR=X",
    "Accenture": "ACN",
    "Microsoft": "MSFT",
}

# ── MODELS ───────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

# ── FUSION PARAMETERS ────────────────────────────────────────────────
TECHNICAL_WEIGHT = 0.5
NEWS_WEIGHT = 0.5

BUY_THRESHOLD = 0.15
SELL_THRESHOLD = -0.15

# both sub-signals must exceed this magnitude, in opposite directions,
# to count as a genuine conflict (vs. one model being weakly neutral)
CONFLICT_MAGNITUDE = 0.2
CONFLICT_CONFIDENCE_CAP = 40

# ── FILE PATHS ───────────────────────────────────────────────────────
DATA_DIR = "data"
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")

os.makedirs(DATA_DIR, exist_ok=True)