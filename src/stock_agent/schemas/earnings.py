"""Earnings context domain model.

Display/context only — the *known-now* next earnings date and proximity. (The ML
model uses a separate, leakage-safe cadence estimate, not this.)
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel


class EarningsContext(BaseModel):
    """Upcoming/last earnings proximity for a ticker as of a date."""

    ticker: str
    as_of: Date
    next_earnings_date: Date | None = None
    last_earnings_date: Date | None = None
    days_to_next_earnings: int | None = None
    days_since_last_earnings: int | None = None
    horizon_days: int | None = None
    # True if the next earnings announcement falls within ``horizon_days``.
    earnings_in_horizon: bool | None = None
