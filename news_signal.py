"""
News signal: pulls sector macro data (indices, currency, competitor),
recent news headlines, filters them for relevance, scores sentiment,
and asks an LLM (Gemini) to turn all of it into a BUY/SELL/HOLD signal.
"""

import json
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
from google import genai
from newsapi import NewsApiClient

import config

_gemini = genai.Client(api_key=config.GEMINI_API_KEY)


# ── MACRO ──────────────────────────────────────────────────────────

def get_macro_data() -> dict:
    macro = {}
    for name, symbol in config.MACRO_TICKERS.items():
        df = yf.download(symbol, period="3d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 2:
            macro[name] = {"price": None, "change_pct": None}
            continue

        prev_close = df["Close"].iloc[-2]
        latest = df["Close"].iloc[-1]
        change_pct = ((latest - prev_close) / prev_close) * 100
        macro[name] = {"price": round(latest, 2), "change_pct": round(change_pct, 2)}

    return macro


def interpret_macro(macro_data: dict):
    """Turn raw macro numbers into (signals list, overall mood, sentiment score)."""
    signals = []
    sentiment = 0

    niftyit = macro_data.get("NIFTY IT Index", {}).get("change_pct")
    if niftyit is not None:
        if niftyit > 1:
            signals.append(("BULLISH", "NIFTY IT Index is strongly up — sector momentum is positive"))
            sentiment += 1
        elif niftyit < -1:
            signals.append(("BEARISH", "NIFTY IT Index is down — sector under broad selling"))
            sentiment -= 1
        else:
            signals.append(("NEUTRAL", "NIFTY IT Index is flat today"))

    usdinr = macro_data.get("USD/INR", {}).get("price")
    usdinr_chg = macro_data.get("USD/INR", {}).get("change_pct")
    if usdinr is not None and usdinr_chg is not None:
        if usdinr_chg > 0.5:
            signals.append((
                "BULLISH",
                f"USD/INR at {usdinr} — strong dollar boosts {config.COMPANY_NAME} revenue "
                f"in rupee terms, up {usdinr_chg}%",
            ))
            sentiment += 1
        elif usdinr_chg < -0.5:
            signals.append((
                "BEARISH",
                f"USD/INR at {usdinr} — rupee strength hurts IT export earnings, "
                f"down {usdinr_chg}%",
            ))
            sentiment -= 1
        else:
            signals.append(("NEUTRAL", f"USD/INR at {usdinr} — currency stable"))

    acn_chg = macro_data.get("Accenture", {}).get("change_pct")
    if acn_chg is not None:
        if acn_chg > 1.5:
            signals.append(("BULLISH", f"Accenture up — positive read-through for {config.COMPANY_NAME} deal pipeline"))
            sentiment += 1
        elif acn_chg < -1.5:
            signals.append(("BEARISH", "Accenture falling — may indicate weak IT services demand"))
            sentiment -= 1
        else:
            signals.append(("NEUTRAL", f"Accenture flat ({acn_chg:+.2f}%)"))

    if sentiment >= 2:
        overall = "BULLISH"
    elif sentiment <= -2:
        overall = "BEARISH"
    else:
        overall = "NEUTRAL"

    return signals, overall, sentiment


# ── NEWS ───────────────────────────────────────────────────────────

def get_news(max_articles: int = 10) -> list:
    newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)

    queries = [
        f"{config.COMPANY_NAME} stock",
        f"{config.COMPANY_NAME} quarterly results deal win contract",
        f"{config.SECTOR} sector outlook",
        f"{config.COMPANY_NAME} attrition hiring revenue forecast",
    ]

    all_articles = []
    seen_titles = set()
    from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    for query in queries:
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            language="en",
            sort_by="publishedAt",
            page_size=10,
        )
        for article in response.get("articles", []):
            title = article.get("title", "")
            if title in seen_titles or title == "[Removed]":
                continue
            seen_titles.add(title)
            all_articles.append({
                "title": title,
                "description": article.get("description", ""),
                "source": article.get("source", {}).get("name", ""),
                "published": article.get("publishedAt", ""),
                "url": article.get("url", ""),
            })

    all_articles.sort(key=lambda x: x["published"], reverse=True)
    return all_articles[:max_articles]


def get_headlines_only(articles: list) -> list:
    headlines = []
    for art in articles:
        title, desc = art["title"], art["description"]
        if desc and desc != title:
            headlines.append(f"{title} — {desc[:100]}")
        else:
            headlines.append(title)
    return headlines


def _clean_json(raw: str) -> str:
    return raw.strip().replace("```json", "").replace("```", "").strip()


def filter_relevant_headlines(headlines: list) -> list:
    """Drop headlines that aren't actually relevant to the ticker/sector."""
    if not headlines:
        return []

    prompt = f"""
You are a financial news filter for {config.COMPANY_NAME} stock analysis.

From the list below, return ONLY the article numbers that are
DIRECTLY or INDIRECTLY relevant to {config.COMPANY_NAME} or {config.SECTOR}.

Remove:
- Articles about unrelated companies or sectors
- Generic economic commentary with no IT angle
- Duplicate or near-duplicate headlines

Headlines:
{chr(10).join(f"{i + 1}. {h}" for i, h in enumerate(headlines))}

Return ONLY a JSON array of relevant index numbers like: [1, 3, 5]
No explanation, no markdown.
"""
    try:
        response = _gemini.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
        indices = json.loads(_clean_json(response.text))
        filtered = [headlines[i - 1] for i in indices if 0 < i <= len(headlines)]
        return filtered
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  Relevance filter failed ({e}) — falling back to unfiltered headlines")
        return headlines


def score_headlines_gemini(headlines: list) -> list:
    if not headlines:
        return []

    numbered = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines))
    prompt = f"""
You are a financial analyst specializing in {config.SECTOR} stocks.

Analyze the sentiment of each headline below specifically for {config.COMPANY_NAME} stock.
Consider: deal wins, revenue impact, macro effects on IT exporters,
USD/INR impact, client spending trends, hiring, margins.

Headlines:
{numbered}

Return ONLY a JSON array like this (no explanation, no markdown):
[
  {{"index": 1, "score": 0.8, "label": "POSITIVE", "reason": "..."}}
]

Rules:
- score range: -1.0 (very negative) to +1.0 (very positive)
- label: POSITIVE (score > 0.1), NEGATIVE (score < -0.1), NEUTRAL (between)
- reason: one line, specific to {config.COMPANY_NAME} and {config.SECTOR} context
"""
    response = _gemini.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)
    data = json.loads(_clean_json(response.text))

    results = []
    for item in data:
        idx = item["index"] - 1
        results.append({
            "headline": headlines[idx] if idx < len(headlines) else "",
            "score": item["score"],
            "label": item["label"],
            "reason": item["reason"],
        })
    return results


def get_sentiment_summary(scored: list) -> dict:
    if not scored:
        return {"overall_score": 0, "label": "NEUTRAL", "positive": 0, "negative": 0,
                "neutral": 0, "total": 0, "strongest_bullish": None, "strongest_bearish": None}

    scores = [s["score"] for s in scored]
    avg_score = round(sum(scores) / len(scores), 3)
    pos = sum(1 for s in scored if s["label"] == "POSITIVE")
    neg = sum(1 for s in scored if s["label"] == "NEGATIVE")
    neut = sum(1 for s in scored if s["label"] == "NEUTRAL")

    if avg_score > 0.1:
        label = "POSITIVE"
    elif avg_score < -0.1:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return {
        "overall_score": avg_score,
        "label": label,
        "positive": pos,
        "negative": neg,
        "neutral": neut,
        "total": len(scored),
        "strongest_bullish": max(scored, key=lambda x: x["score"]),
        "strongest_bearish": min(scored, key=lambda x: x["score"]),
    }


# ── COMBINED NEWS + MACRO SIGNAL ────────────────────────────────────

def _build_prompt(macro_data, macro_signals, summary, scored_headlines) -> str:
    macro_text = ""
    for name, data in macro_data.items():
        price, change = data.get("price"), data.get("change_pct")
        if price is None:
            macro_text += f"  - {name}: Unavailable\n"
        else:
            direction = "UP" if change >= 0 else "DOWN"
            macro_text += f"  - {name}: {price} ({direction} {abs(change)}%)\n"

    signals_text = "".join(f"  - [{tag}] {msg}\n" for tag, msg in macro_signals)

    news_text = ""
    for i, item in enumerate(scored_headlines, 1):
        news_text += (
            f"  {i}. [{item['label']}] Score: {item['score']:+.2f}\n"
            f"     Headline : {item['headline'][:100]}\n"
            f"     Reason   : {item['reason']}\n\n"
        )

    sent_breakdown = (
        f"{summary['positive']} positive, {summary['negative']} negative, "
        f"{summary['neutral']} neutral out of {summary['total']} articles"
    )

    return f"""
You are a senior Indian stock market analyst specializing in the {config.SECTOR} sector.
Today's date: {datetime.now().strftime('%d %B %Y')}

Your job is to give a short-term (1-5 day) trading signal for {config.COMPANY_NAME} stock
listed on NSE, based on current news sentiment and macro conditions.

MACRO CONDITIONS
{macro_text}
MACRO SIGNALS INTERPRETED:
{signals_text}

NEWS SENTIMENT ANALYSIS
Overall Sentiment Score : {summary['overall_score']:+.3f} ({summary['label']})
Breakdown               : {sent_breakdown}

Individual Headlines:
{news_text}

TASK
Based on ALL the above, provide your analysis in this EXACT JSON format
(no markdown, no explanation outside JSON):

{{
  "signal": "BUY" or "SELL" or "HOLD",
  "confidence": a number between 0 and 100,
  "reasoning": "3-4 lines explaining the signal clearly",
  "key_positives": ["point 1", "point 2"],
  "key_risks": ["risk 1", "risk 2"]
}}
"""


def get_news_signal() -> dict:
    """Full news+macro pipeline in one call. Returns a signal dict shaped
    like technical_signal.get_technical_signal()'s output, for easy fusion."""
    macro_data = get_macro_data()
    macro_signals, _, _ = interpret_macro(macro_data)

    articles = get_news()
    headlines = get_headlines_only(articles)
    filtered = filter_relevant_headlines(headlines)
    scored = score_headlines_gemini(filtered)
    summary = get_sentiment_summary(scored)

    prompt = _build_prompt(macro_data, macro_signals, summary, scored)
    response = _gemini.models.generate_content(model=config.GEMINI_MODEL, contents=prompt)

    try:
        data = json.loads(_clean_json(response.text))
    except json.JSONDecodeError as e:
        print(f"  News signal JSON parse failed ({e}) — defaulting to HOLD")
        data = {"signal": "HOLD", "confidence": 0, "reasoning": "Parse error", "key_positives": [], "key_risks": []}

    return {
        "ticker": config.TICKER,
        "signal": data.get("signal", "HOLD"),
        "confidence": data.get("confidence", 0),
        "reasoning": [data.get("reasoning", "")] if isinstance(data.get("reasoning"), str) else data.get("reasoning", []),
        "risk": None,  # news pipeline doesn't rate its own risk; fusion decides this
        "key_positives": data.get("key_positives", []),
        "key_risks": data.get("key_risks", []),
        "raw": response.text,
    }
