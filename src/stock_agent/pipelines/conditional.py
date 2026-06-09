"""Conditional event-study pipeline: load two tickers and run the study (#7).

Thin orchestration — fetch target + driver price history via the provider
registry, convert to date-indexed close series, and delegate to the pure
``analysis.conditional`` function. No business logic here.
"""

from __future__ import annotations

import pandas as pd

from stock_agent.analysis.conditional import conditional_forward_returns
from stock_agent.data.loader import PriceLoader
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.schemas.conditional import ConditionalStudy, Direction
from stock_agent.settings import Settings

# ~7 years of daily history so even rare shocks accumulate a usable event count.
_LOOKBACK_DAYS = 365 * 7


def _close_series(loader: PriceLoader, ticker: str, min_bars: int) -> pd.Series:
    """Load recent prices for ``ticker`` as a date-indexed close-price Series."""
    series = loader.load_recent(ticker.upper(), _LOOKBACK_DAYS, min_bars=min_bars).series
    return pd.Series(series.closes, index=pd.to_datetime(series.dates))


def run_conditional_study(
    target: str,
    driver: str,
    *,
    shock_pct: float = 0.05,
    event_window_days: int = 5,
    horizon_days: int = 10,
    direction: Direction = "both",
    settings: Settings,
    registry: ProviderRegistry | None = None,
) -> ConditionalStudy:
    """Load target + driver prices and compute the conditional forward-return study."""
    registry = registry or build_default_registry(settings)
    loader = PriceLoader(registry)
    # Need at least a trailing window + a forward horizon, plus headroom for events.
    min_bars = event_window_days + horizon_days + 30
    target_closes = _close_series(loader, target, min_bars)
    driver_closes = _close_series(loader, driver, min_bars)
    return conditional_forward_returns(
        target_closes,
        driver_closes,
        target=target.upper(),
        driver=driver.upper(),
        shock_pct=shock_pct,
        event_window=event_window_days,
        horizon=horizon_days,
        direction=direction,
    )
