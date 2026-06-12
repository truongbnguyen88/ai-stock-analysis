"""Market-wide context series (e.g. VIX) for macro / volatility features.

These series are the SAME for every ticker on a given date — a *market state*,
not a cross-sectional signal. So they help the volatility / big-move target far
more than direction (which is cross-sectional and ~efficient). See
docs/validations_results.md.

Point-in-time safety: VIX is a real-time index (no publication lag) → the value
on date ``t`` is known at ``t``. Macro series with release lags (employment, CPI)
are intentionally NOT here yet — they need release-date (vintage) alignment to
avoid leakage.
"""

from __future__ import annotations

from datetime import date as Date

import pandas as pd

from stock_agent.logging_config import get_logger
from stock_agent.providers.base import ProviderError
from stock_agent.providers.registry import ProviderRegistry

log = get_logger(__name__)

VIX_SYMBOL = "^VIX"  # CBOE Volatility Index (yfinance ticker)
MARKET_SYMBOL = "SPY"  # S&P 500 ETF — broad-market proxy for relative-strength features


def fetch_index_close(
    registry: ProviderRegistry, symbol: str, *, start: Date, end: Date
) -> pd.Series:
    """Return a date-indexed close series for a market index (empty on failure).

    Degrades gracefully: on any provider error returns an empty Series, so the
    dependent features simply become NaN rather than breaking the pipeline.
    """
    try:
        series = registry.get_daily_prices(symbol, start, end)
    except ProviderError as exc:
        log.warning("market_context.fetch_failed", symbol=symbol, error=str(exc))
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex([b.date for b in series.bars])
    return pd.Series([b.close for b in series.bars], index=idx, dtype=float)


def fetch_vix(registry: ProviderRegistry, *, start: Date, end: Date) -> pd.Series:
    """Date-indexed VIX close over [start, end] (empty Series if unavailable)."""
    return fetch_index_close(registry, VIX_SYMBOL, start=start, end=end)


def fetch_market(registry: ProviderRegistry, *, start: Date, end: Date) -> pd.Series:
    """Date-indexed broad-market (SPY) close over [start, end] for relative strength.

    Like VIX, fetched once over a span and reindexed per fold; using close at-or-
    before each date keeps relative-strength features point-in-time safe.
    """
    return fetch_index_close(registry, MARKET_SYMBOL, start=start, end=end)
