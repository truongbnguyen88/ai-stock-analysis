"""Provider abstraction — interfaces and the uniform error contract.

Concrete providers (yfinance, Finnhub, Marketaux, Alpha Vantage) implement these
structural Protocols and normalize their raw API responses into the shared
``schemas`` types. Business logic depends only on these Protocols, never on a
specific provider, so providers can be added/removed via configuration alone.

Error contract: every provider raises one of the typed errors below. The
registry catches them to drive fallback; callers get a predictable surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date
from enum import StrEnum
from typing import Protocol, runtime_checkable

from stock_agent.schemas.market import Fundamentals, PriceSeries
from stock_agent.schemas.news import NewsBundle


class Capability(StrEnum):
    """A data capability a provider may offer. Used to resolve fallback chains."""

    PRICES = "prices"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


# ---------------------------------------------------------------------------
# Error contract
# ---------------------------------------------------------------------------
class ProviderError(RuntimeError):
    """Base class for all provider failures. Carries the offending provider name."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.message = message


class ProviderUnavailable(ProviderError):
    """Transient/config failure (missing key, network error, 5xx, bad auth).

    Signals the registry to fall through to the next provider in the chain.
    """


class ProviderRateLimit(ProviderUnavailable):
    """Rate limit hit (HTTP 429 or provider-specific quota message).

    A subtype of ``ProviderUnavailable`` so the registry treats it as fall-through.
    """


class SymbolNotFound(ProviderError):
    """The requested ticker is unknown to this provider, or has no data."""


# ---------------------------------------------------------------------------
# Capability Protocols (structural typing — no inheritance required by impls)
# ---------------------------------------------------------------------------
@runtime_checkable
class Provider(Protocol):
    """Common surface every provider exposes (identity + availability).

    The capability Protocols below extend this so the registry can treat any
    provider uniformly for naming/availability while still narrowing to a
    specific capability via ``isinstance``.
    """

    name: str

    def available(self) -> bool:
        """Whether this provider can be used now (e.g. required key present)."""
        ...


@runtime_checkable
class PriceProvider(Provider, Protocol):
    """Supplies daily OHLCV price history."""

    def get_daily_prices(self, ticker: str, start: Date, end: Date) -> PriceSeries:
        """Return daily bars for ``ticker`` within [start, end] (inclusive)."""
        ...


@runtime_checkable
class NewsProvider(Provider, Protocol):
    """Supplies company-related news articles."""

    def get_company_news(self, ticker: str, start: Date, end: Date) -> NewsBundle:
        """Return articles for ``ticker`` published within [start, end]."""
        ...


@runtime_checkable
class TopicNewsProvider(Provider, Protocol):
    """Supplies theme/keyword-scoped news (not ticker-scoped) — Enhancement C.

    Takes the resolved ``keywords`` (phrases) and builds its OWN native query
    syntax internally, so heterogeneous providers (GDELT ``OR`` vs Marketaux
    ``|``) can share the chain. Distinct from the symbol-scoped
    ``NewsProvider.get_company_news``.
    """

    def get_topic_news(
        self, keywords: Sequence[str], start: Date, end: Date, *, top_n: int = 25
    ) -> NewsBundle:
        """Return newest-first articles matching any of ``keywords`` within [start, end]."""
        ...


@runtime_checkable
class FundamentalsProvider(Provider, Protocol):
    """Supplies company fundamentals.

    Protocol defined here for completeness; the first concrete implementation
    lands in Phase 4 when the report's fundamentals section needs it.
    """

    def get_fundamentals(self, ticker: str) -> Fundamentals: ...


@runtime_checkable
class EarningsProvider(Provider, Protocol):
    """Supplies earnings announcement dates (past and upcoming)."""

    def get_earnings_dates(self, ticker: str) -> list[Date]:
        """Return known earnings announcement dates for ``ticker``, sorted ascending."""
        ...
