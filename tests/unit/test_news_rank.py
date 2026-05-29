"""Tests for relevance ranking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stock_agent.news.rank import rank_articles
from stock_agent.schemas.news import Article

_NOW = datetime(2025, 2, 1, tzinfo=UTC)


def _article(title: str, age_days: int, *, summary: str | None = None) -> Article:
    return Article(
        title=title,
        url=f"https://x.com/{title.replace(' ', '-')}-{age_days}",
        source="s",
        published_at=_NOW - timedelta(days=age_days),
        summary=summary,
    )


def test_recency_orders_results() -> None:
    arts = [_article("NVDA old", 25), _article("NVDA new", 1)]
    ranked = rank_articles(arts, "NVDA", now=_NOW, lookback_days=30)
    assert ranked[0].title == "NVDA new"


def test_ticker_mention_boosts_relevance() -> None:
    mentions = _article("NVDA earnings beat", 10)
    generic = _article("Market wrap", 10)
    ranked = rank_articles([generic, mentions], "NVDA", now=_NOW, lookback_days=30)
    assert ranked[0].title == "NVDA earnings beat"
    assert all(0.0 <= (a.relevance or -1) <= 1.0 for a in ranked)


def test_top_n_truncates() -> None:
    arts = [_article(f"NVDA item {i}", i) for i in range(10)]
    ranked = rank_articles(arts, "NVDA", now=_NOW, lookback_days=30, top_n=3)
    assert len(ranked) == 3


def test_naive_published_at_handled() -> None:
    naive = Article(
        title="NVDA naive", url="https://x.com/n", source="s", published_at=datetime(2025, 1, 31)
    )
    ranked = rank_articles([naive], "NVDA", now=_NOW, lookback_days=30)
    assert ranked[0].relevance is not None
