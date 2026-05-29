"""Price loading orchestration.

Thin layer over the provider registry: fetch a ``PriceSeries`` (with provider
fallback handled by the registry), run non-fatal data-quality validation, and
return both together. Keeps business logic out of the providers and the CLI.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta

from pydantic import BaseModel

from stock_agent.data.validation import DataIssue, validate_price_series
from stock_agent.logging_config import get_logger
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)


class LoadResult(BaseModel):
    """A loaded price series plus any data-quality issues found."""

    series: PriceSeries
    issues: list[DataIssue]


class PriceLoader:
    """Loads validated price history via the provider registry."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def load(
        self,
        ticker: str,
        start: Date,
        end: Date,
        *,
        min_bars: int = 20,
    ) -> LoadResult:
        """Fetch [start, end] daily prices and validate them."""
        series = self._registry.get_daily_prices(ticker, start, end)
        issues = validate_price_series(series, min_bars=min_bars, today=end)
        if issues:
            log.info(
                "loader.data_issues",
                ticker=ticker,
                n_issues=len(issues),
                codes=[i.code for i in issues],
            )
        return LoadResult(series=series, issues=issues)

    def load_recent(
        self,
        ticker: str,
        lookback_days: int,
        *,
        end: Date | None = None,
        min_bars: int = 20,
    ) -> LoadResult:
        """Convenience: load the last ``lookback_days`` calendar days up to ``end``.

        Note: ``lookback_days`` is a calendar window. Callers needing N *trading*
        days (e.g. MA200) should pass a larger window to absorb weekends/holidays.
        """
        end = end or Date.today()
        start = end - timedelta(days=lookback_days)
        return self.load(ticker, start, end, min_bars=min_bars)
