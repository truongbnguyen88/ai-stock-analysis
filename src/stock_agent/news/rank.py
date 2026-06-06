"""Relevance ranking for news articles.

A transparent, deterministic heuristic (no LLM): relevance combines recency,
whether the ticker/company is mentioned (title weighted over summary), and
whether a summary exists at all. Scores are in [0, 1]; articles are returned
sorted by relevance (then recency), optionally truncated to ``top_n``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from stock_agent.schemas.news import Article

Order = Literal["relevance", "recency"]

# Component weights (sum to 1.0).
_W_RECENCY = 0.5
_W_MENTION = 0.4
_W_SUMMARY = 0.1


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _recency_score(published: datetime, now: datetime, lookback_days: int) -> float:
    """Linear decay from 1.0 (now) to 0.0 at the lookback horizon."""
    age_days = (now - _as_utc(published)).total_seconds() / 86_400.0
    if age_days <= 0:
        return 1.0
    if age_days >= lookback_days:
        return 0.0
    return 1.0 - age_days / lookback_days


def _mention_score(article: Article, ticker: str, company_name: str | None) -> float:
    """1.0 if ticker/company in title, 0.6 if only in summary, else 0.0."""
    needles = [ticker.lower()] + ([company_name.lower()] if company_name else [])
    title = article.title.lower()
    summary = (article.summary or "").lower()
    if any(n in title for n in needles):
        return 1.0
    if any(n in summary for n in needles):
        return 0.6
    return 0.0


def rank_articles(
    articles: list[Article],
    ticker: str,
    *,
    company_name: str | None = None,
    now: datetime | None = None,
    lookback_days: int = 30,
    top_n: int | None = None,
    order: Order = "relevance",
) -> list[Article]:
    """Score, sort (desc), and optionally truncate articles.

    ``order="relevance"`` (default) ranks by the relevance heuristic then recency — best
    for synthesis/summarization (filters off-topic items). ``order="recency"`` ranks
    strictly newest-first (relevance as tiebreak) — for "latest news" pulls where freshness
    is the priority. Relevance is always computed and attached either way.
    """
    now = now or datetime.now(UTC)
    scored: list[Article] = []
    for article in articles:
        score = (
            _W_RECENCY * _recency_score(article.published_at, now, lookback_days)
            + _W_MENTION * _mention_score(article, ticker, company_name)
            + _W_SUMMARY * (1.0 if article.summary else 0.0)
        )
        scored.append(article.model_copy(update={"relevance": round(score, 4)}))

    if order == "recency":  # newest first; relevance breaks ties
        scored.sort(key=lambda a: (_as_utc(a.published_at), a.relevance or 0.0), reverse=True)
    else:  # relevance first; recency breaks ties
        scored.sort(key=lambda a: (a.relevance or 0.0, _as_utc(a.published_at)), reverse=True)
    return scored[:top_n] if top_n is not None else scored
