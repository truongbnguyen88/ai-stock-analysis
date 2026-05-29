"""In-memory fake provider for offline, deterministic tests.

Returns injected data and can be configured to fail with a specific error, so
tests can exercise the registry's fallback/merge logic without any network. It
structurally satisfies all three capability Protocols.
"""

from __future__ import annotations

from datetime import date as Date

from stock_agent.providers.base import ProviderError
from stock_agent.schemas.market import Fundamentals, PriceSeries
from stock_agent.schemas.news import NewsBundle


class FakeProvider:
    """A configurable test double for price/news/fundamentals capabilities."""

    def __init__(
        self,
        name: str,
        *,
        prices: PriceSeries | None = None,
        news: NewsBundle | None = None,
        fundamentals: Fundamentals | None = None,
        available: bool = True,
        raises: ProviderError | None = None,
    ) -> None:
        self.name = name
        self._prices = prices
        self._news = news
        self._fundamentals = fundamentals
        self._available = available
        self._raises = raises  # if set, every capability call raises this

    def available(self) -> bool:
        return self._available

    def get_daily_prices(self, ticker: str, start: Date, end: Date) -> PriceSeries:
        if self._raises is not None:
            raise self._raises
        if self._prices is None:
            raise ProviderError(self.name, "no prices configured")
        return self._prices

    def get_company_news(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        if self._raises is not None:
            raise self._raises
        if self._news is None:
            raise ProviderError(self.name, "no news configured")
        return self._news

    def get_fundamentals(self, ticker: str) -> Fundamentals:
        if self._raises is not None:
            raise self._raises
        if self._fundamentals is None:
            raise ProviderError(self.name, "no fundamentals configured")
        return self._fundamentals
