"""PriceLoader end-to-end over a fake provider registry (no network)."""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.data.loader import PriceLoader
from stock_agent.providers.base import SymbolNotFound
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.settings import Settings


def _registry(provider: FakeProvider) -> ProviderRegistry:
    return ProviderRegistry(
        [provider], Settings(_env_file=None, provider_price_priority=provider.name)
    )


def test_load_returns_series_and_quality_issues() -> None:
    series = PriceSeries(
        ticker="X", bars=[PriceBar(date=date(2025, 1, 2), open=1.0, high=2.0, low=0.5, close=1.5)]
    )
    loader = PriceLoader(_registry(FakeProvider("yfinance", prices=series)))
    result = loader.load("X", date(2025, 1, 1), date(2025, 1, 2), min_bars=20)
    assert result.series.ticker == "X"
    # Single bar < min_bars -> flagged but non-fatal.
    assert any(i.code == "short_history" for i in result.issues)


def test_load_propagates_provider_error() -> None:
    loader = PriceLoader(
        _registry(FakeProvider("yfinance", raises=SymbolNotFound("yfinance", "no data")))
    )
    with pytest.raises(SymbolNotFound):
        loader.load("BAD", date(2025, 1, 1), date(2025, 1, 2))
