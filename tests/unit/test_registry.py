"""Registry fallback (prices) and merge/availability (news) logic."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from stock_agent.providers.base import ProviderUnavailable, SymbolNotFound
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings


def _series(ticker: str) -> PriceSeries:
    bar = PriceBar(date=date(2025, 1, 2), open=1.0, high=2.0, low=0.5, close=1.5)
    return PriceSeries(ticker=ticker, bars=[bar])


def _news(ticker: str, url: str) -> NewsBundle:
    art = Article(title="t", url=url, source="s", published_at=datetime(2025, 1, 2, tzinfo=UTC))
    return NewsBundle(ticker=ticker, articles=[art])


def _settings(price: str = "", news: str = "") -> Settings:
    return Settings(_env_file=None, provider_price_priority=price, provider_news_priority=news)


def test_prices_failover_to_second_provider() -> None:
    p1 = FakeProvider("p1", raises=ProviderUnavailable("p1", "down"))
    p2 = FakeProvider("p2", prices=_series("X"))
    reg = ProviderRegistry([p1, p2], _settings(price="p1,p2"))
    assert reg.get_daily_prices("X", date(2025, 1, 1), date(2025, 1, 2)).ticker == "X"


def test_prices_all_fail_raises_last_error() -> None:
    p1 = FakeProvider("p1", raises=ProviderUnavailable("p1", "down"))
    p2 = FakeProvider("p2", raises=SymbolNotFound("p2", "missing"))
    reg = ProviderRegistry([p1, p2], _settings(price="p1,p2"))
    with pytest.raises(SymbolNotFound):
        reg.get_daily_prices("X", date(2025, 1, 1), date(2025, 1, 2))


def test_prices_no_available_providers_raises() -> None:
    reg = ProviderRegistry([FakeProvider("p1", prices=_series("X"))], _settings(price="other"))
    with pytest.raises(ProviderUnavailable):
        reg.get_daily_prices("X", date(2025, 1, 1), date(2025, 1, 2))


def test_news_merge_combines_all_providers() -> None:
    n1 = FakeProvider("n1", news=_news("X", "https://a.com/1"))
    n2 = FakeProvider("n2", news=_news("X", "https://b.com/2"))
    reg = ProviderRegistry([n1, n2], _settings(news="n1,n2"))
    bundle = reg.get_company_news("X", date(2025, 1, 1), date(2025, 1, 31))
    assert len(bundle) == 2


def test_news_failover_returns_first_only() -> None:
    n1 = FakeProvider("n1", news=_news("X", "https://a.com/1"))
    n2 = FakeProvider("n2", news=_news("X", "https://b.com/2"))
    reg = ProviderRegistry([n1, n2], _settings(news="n1,n2"))
    bundle = reg.get_company_news("X", date(2025, 1, 1), date(2025, 1, 31), mode="failover")
    assert len(bundle) == 1


def test_news_skips_unavailable_provider() -> None:
    n1 = FakeProvider("n1", news=_news("X", "https://a.com/1"), available=False)
    n2 = FakeProvider("n2", news=_news("X", "https://b.com/2"))
    reg = ProviderRegistry([n1, n2], _settings(news="n1,n2"))
    bundle = reg.get_company_news("X", date(2025, 1, 1), date(2025, 1, 31))
    assert len(bundle) == 1  # n1 filtered out by availability
