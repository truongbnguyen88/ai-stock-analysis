"""NewsFetcher end-to-end over a fake registry (clean -> dedup -> rank)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from stock_agent.news.fetch import NewsFetcher
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings


def _now() -> datetime:
    return datetime.now(UTC)


def test_fetch_cleans_dedups_and_ranks() -> None:
    # Two providers return overlapping stories (one exact-dup story, plus noise).
    recent = _now() - timedelta(days=1)
    older = _now() - timedelta(days=20)
    p1 = FakeProvider(
        "finnhub",
        news=NewsBundle(
            ticker="NVDA",
            articles=[
                Article(
                    title="NVDA beats earnings",
                    url="https://a.com/1?utm_source=x",
                    source="finnhub",
                    published_at=recent,
                    summary="strong quarter",
                ),
                Article(
                    title="Broad market wrap",
                    url="https://a.com/2",
                    source="finnhub",
                    published_at=older,
                ),
            ],
        ),
    )
    p2 = FakeProvider(
        "marketaux",
        news=NewsBundle(
            ticker="NVDA",
            articles=[
                # same story as a.com/1 (canonical URL match after stripping utm)
                Article(
                    title="NVDA beats earnings",
                    url="https://a.com/1",
                    source="marketaux",
                    published_at=recent,
                ),
            ],
        ),
    )
    registry = ProviderRegistry(
        [p1, p2], Settings(_env_file=None, provider_news_priority="finnhub,marketaux")
    )

    bundle = NewsFetcher(registry).fetch("NVDA", lookback_days=30, top_n=10)

    # 3 raw -> 2 after dedup; NVDA-mention + recent ranks first.
    assert len(bundle) == 2
    assert bundle.articles[0].title == "NVDA beats earnings"
    assert bundle.articles[0].relevance is not None
