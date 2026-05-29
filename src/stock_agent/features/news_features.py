"""News feature engineering with Claude batch sentiment scoring.

Sentiment design note:
  Finnhub and Marketaux do not return sentiment scores; only Alpha Vantage
  does, covering ~17% of articles. To get full-coverage, consistent sentiment,
  we send all articles to Claude in a single batched call (~$0.08 per run at
  Sonnet pricing) and receive a score per article in [-1, 1]. This lives in
  ``features/`` (not ``news/``) so it only fires when building the ML feature
  matrix — not on every news fetch for the report or chat agent.

  Limitation: we only have *current* news from the API, not historical news at
  each past date t. Therefore news features are used for live inference only;
  the training matrix uses price features alone (see assembler.py). When
  historical news data becomes available, these functions slot in unchanged.
"""

from __future__ import annotations

import json
import re

from stock_agent.llm.client import TextLLM
from stock_agent.news.clean import canonical_url
from stock_agent.schemas.news import Article, NewsBundle

NEWS_FEATURE_COLS: list[str] = [
    "article_count",
    "avg_sentiment",
    "sentiment_coverage",
    "pct_positive",
    "pct_negative",
    "has_earnings",
    "has_regulatory",
    "has_upgrade",
    "has_downgrade",
]

_SENTIMENT_SYSTEM = """Score each article's financial sentiment for the given ticker.
Return ONLY a JSON object: {"scores": [{"url": "...", "score": 0.42}, ...]}
Scores: -1.0 (very bearish) to +1.0 (very bullish). 0.0 = neutral.
Base scores solely on the article title and summary provided."""

# Keyword patterns for event-type flags (title matching only — efficient + reproducible).
_EARNINGS_RE = re.compile(r"\b(earnings?|eps|beat|miss(ed)?|revenue|quarter|guidance)\b", re.I)
_REGULATORY_RE = re.compile(
    r"\b(sec|ftc|doj|antitrust|regulat|probe|investigation|fine|lawsuit)\b", re.I
)
_UPGRADE_RE = re.compile(r"\b(upgrade[sd]?|raised?|reiterat|outperform|overweight|buy)\b", re.I)
_DOWNGRADE_RE = re.compile(
    r"\b(downgrade[sd]?|lowered?|underperform|underweight|sell|reduce)\b", re.I
)


def _score_sentiment_batch(articles: list[Article], ticker: str, llm: TextLLM) -> dict[str, float]:
    """Send all articles to Claude in one call; return {canonical_url → score}."""
    if not articles:
        return {}

    user_lines = [f"Ticker: {ticker}\n"]
    for i, a in enumerate(articles, 1):
        line = f"[{i}] url: {a.url}\n    title: {a.title}"
        if a.summary:
            line += f"\n    summary: {a.summary[:300]}"  # cap per-article tokens
        user_lines.append(line)

    raw = llm.complete_json(system=_SENTIMENT_SYSTEM, user="\n".join(user_lines), max_tokens=2048)

    try:
        parsed = json.loads(raw)
        items = parsed.get("scores", []) if isinstance(parsed, dict) else []
    except (json.JSONDecodeError, AttributeError):
        return {}

    return {
        canonical_url(str(item["url"])): float(item["score"])
        for item in items
        if isinstance(item, dict) and "url" in item and "score" in item
    }


def build_news_features(
    bundle: NewsBundle,
    llm: TextLLM | None = None,
) -> dict[str, float]:
    """Build the news feature dict for the current article bundle.

    When ``llm`` is provided, Claude scores sentiment for all articles.
    When absent (offline / no key), sentiment features default to 0 and
    ``sentiment_coverage`` to 0 so the ML model learns to ignore them.
    """
    articles = bundle.articles
    n = len(articles)
    if n == 0:
        return {col: 0.0 for col in NEWS_FEATURE_COLS}

    # Sentiment scoring.
    scores: dict[str, float] = {}
    if llm is not None:
        scores = _score_sentiment_batch(articles, bundle.ticker, llm)

    # Map scores back to articles by canonical URL.
    article_scores = [scores.get(canonical_url(str(a.url))) for a in articles]
    scored = [s for s in article_scores if s is not None]

    avg_sentiment = sum(scored) / len(scored) if scored else 0.0
    sentiment_coverage = len(scored) / n

    pos_threshold, neg_threshold = 0.15, -0.15
    pct_positive = sum(1 for s in scored if s > pos_threshold) / n
    pct_negative = sum(1 for s in scored if s < neg_threshold) / n

    # Event flags from title keyword matching.
    titles = " ".join(a.title for a in articles)
    return {
        "article_count": float(n),
        "avg_sentiment": avg_sentiment,
        "sentiment_coverage": sentiment_coverage,
        "pct_positive": pct_positive,
        "pct_negative": pct_negative,
        "has_earnings": float(bool(_EARNINGS_RE.search(titles))),
        "has_regulatory": float(bool(_REGULATORY_RE.search(titles))),
        "has_upgrade": float(bool(_UPGRADE_RE.search(titles))),
        "has_downgrade": float(bool(_DOWNGRADE_RE.search(titles))),
    }
