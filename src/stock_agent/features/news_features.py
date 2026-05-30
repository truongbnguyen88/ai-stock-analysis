"""News display/context features (NOT ML model inputs).

Option A (see docs/TASKS.md): the ML forecaster is PRICE-ONLY. News sentiment is
never a model feature — it is shown alongside the forecast as context. We have no
point-in-time historical news to train on, so a news feature could not be applied
at inference consistently anyway.

Sentiment sourcing (display only):
  - default: Alpha Vantage's free pre-computed ``article.sentiment`` (~17% of
    articles carry a score; ``sentiment_coverage`` reports the fraction).
  - optional: Claude batch scoring for full coverage (off by default; ~$0.035 per
    25 articles, measured — see scripts/estimate_sentiment_cost.py). Scoring all
    ~300 articles costs ~$0.44 and is fragile (truncates), so the default path
    uses AV + the existing Role A summary for insight instead.
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


def build_sentiment_user(articles: list[Article], ticker: str) -> str:
    """Build the user message for batch sentiment scoring (one line block per article)."""
    user_lines = [f"Ticker: {ticker}\n"]
    for i, a in enumerate(articles, 1):
        line = f"[{i}] url: {a.url}\n    title: {a.title}"
        if a.summary:
            line += f"\n    summary: {a.summary[:300]}"  # cap per-article tokens
        user_lines.append(line)
    return "\n".join(user_lines)


def _score_sentiment_batch(articles: list[Article], ticker: str, llm: TextLLM) -> dict[str, float]:
    """Send all articles to Claude in one call; return {canonical_url → score}."""
    if not articles:
        return {}

    user = build_sentiment_user(articles, ticker)
    raw = llm.complete_json(system=_SENTIMENT_SYSTEM, user=user, max_tokens=2048)

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
    *,
    llm: TextLLM | None = None,
    use_llm_sentiment: bool = False,
) -> dict[str, float]:
    """Build the news **display/context** dict for the current article bundle.

    NOTE (Option A): this is NOT a model input — the ML forecaster is price-only.
    These features are shown alongside the forecast (report / agent) as context.

    Sentiment source:
      - default: Alpha Vantage's pre-computed ``article.sentiment`` (free; ~17%
        coverage — ``sentiment_coverage`` reports how much).
      - opt-in: pass ``use_llm_sentiment=True`` with an ``llm`` to score ALL
        articles with Claude (full coverage, ~$0.04/25 articles; see the cost
        decision in docs/TASKS.md). Off by default to avoid per-call cost.
    """
    articles = bundle.articles
    n = len(articles)
    if n == 0:
        return {col: 0.0 for col in NEWS_FEATURE_COLS}

    # Sentiment per article: Claude (opt-in) overrides AV's free pre-computed score.
    if use_llm_sentiment and llm is not None:
        scores = _score_sentiment_batch(articles, bundle.ticker, llm)
        article_scores = [scores.get(canonical_url(str(a.url))) for a in articles]
    else:
        article_scores = [a.sentiment for a in articles]  # AV-provided (None if absent)
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
