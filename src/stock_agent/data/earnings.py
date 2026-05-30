"""Earnings date helpers: context (display) and cadence (for the ML feature).

Two distinct uses, deliberately computed differently:
- ``earnings_context`` uses the *known-now* dates (the real upcoming announcement)
  → display context for the report and agent.
- ``earnings_cadence_days`` + ``days_to_next_earnings_estimate`` use only *past*
  dates and the quarterly cadence → a leakage-safe ML feature (point-in-time
  valid; the real future date isn't reliably known weeks ahead).
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import date as Date

from stock_agent.logging_config import get_logger
from stock_agent.providers.base import ProviderError
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.earnings import EarningsContext

log = get_logger(__name__)

_DEFAULT_CADENCE_DAYS = 91.0  # quarterly
_MIN_GAP, _MAX_GAP = 30, 200  # plausible quarterly spacing (filters yfinance outliers)


def earnings_context(
    dates: Sequence[Date], *, ticker: str, as_of: Date, horizon_days: int
) -> EarningsContext:
    """Build display context from known earnings dates (uses the real next date)."""
    past = [d for d in dates if d <= as_of]
    future = [d for d in dates if d > as_of]
    next_date = future[0] if future else None
    last_date = past[-1] if past else None
    days_to_next = (next_date - as_of).days if next_date is not None else None
    return EarningsContext(
        ticker=ticker.upper(),
        as_of=as_of,
        next_earnings_date=next_date,
        last_earnings_date=last_date,
        days_to_next_earnings=days_to_next,
        days_since_last_earnings=(as_of - last_date).days if last_date is not None else None,
        horizon_days=horizon_days,
        earnings_in_horizon=(days_to_next <= horizon_days) if days_to_next is not None else None,
    )


def fetch_earnings_context(
    registry: ProviderRegistry, ticker: str, *, as_of: Date, horizon_days: int
) -> EarningsContext:
    """Fetch dates via the registry and build context; degrade to empty on failure."""
    try:
        dates = registry.get_earnings_dates(ticker)
    except ProviderError as exc:
        log.warning("earnings.fetch_failed", ticker=ticker, error=str(exc))
        return EarningsContext(ticker=ticker.upper(), as_of=as_of, horizon_days=horizon_days)
    return earnings_context(dates, ticker=ticker, as_of=as_of, horizon_days=horizon_days)


def earnings_cadence_days(dates: Sequence[Date]) -> float:
    """Median spacing between consecutive earnings dates (default quarterly).

    Filters to plausible quarterly gaps so a missing/duplicated yfinance row does
    not distort the cadence.
    """
    ordered = sorted(dates)
    if len(ordered) < 2:
        return _DEFAULT_CADENCE_DAYS
    gaps = [(b - a).days for a, b in zip(ordered, ordered[1:], strict=False)]
    plausible = [g for g in gaps if _MIN_GAP <= g <= _MAX_GAP]
    return float(statistics.median(plausible or gaps))


def days_to_next_earnings_estimate(
    as_of: Date, past_dates: Sequence[Date], cadence: float
) -> float | None:
    """Leakage-safe estimate of days-to-next-earnings using ONLY past dates.

    next ≈ last_past_earnings + cadence; estimate = clip(next - as_of, 0, cadence).
    Uses no future information, so it is valid as a point-in-time training feature.
    Returns None if there is no prior earnings date.
    """
    past = [d for d in past_dates if d <= as_of]
    if not past:
        return None
    last = max(past)
    days_since = (as_of - last).days
    estimate = cadence - days_since
    # Clamp to [0, cadence]: if we're "past due" vs cadence, the event is imminent.
    return float(min(cadence, max(0.0, estimate)))
