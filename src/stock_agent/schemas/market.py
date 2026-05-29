"""Market-data domain models: prices and fundamentals.

These are the normalized shapes providers map their raw API responses into, so
downstream code (indicators, features, reports) never depends on a specific
provider's JSON layout.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Self

from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, model_validator


class PriceBar(BaseModel):
    """A single daily OHLCV bar.

    ``adj_close`` is the split/dividend-adjusted close when the provider supplies
    it; analytics needing total-return continuity should prefer it over ``close``.
    """

    date: Date
    open: NonNegativeFloat
    high: NonNegativeFloat
    low: NonNegativeFloat
    close: NonNegativeFloat
    adj_close: NonNegativeFloat | None = None
    volume: NonNegativeInt = 0

    @model_validator(mode="after")
    def _check_ohlc(self) -> Self:
        # Basic OHLC sanity: high must bound low and the open/close prints.
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low}) on {self.date}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open {self.open} outside [low, high] on {self.date}")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close {self.close} outside [low, high] on {self.date}")
        return self


class PriceSeries(BaseModel):
    """An ordered daily price history for one ticker.

    Invariant: ``bars`` are strictly date-ascending with no duplicates — enforced
    here so indicator/feature code can assume correct point-in-time ordering.
    """

    ticker: str
    bars: list[PriceBar]

    @model_validator(mode="after")
    def _check_sorted_unique(self) -> Self:
        dates = [b.date for b in self.bars]
        # strict=False: dates[1:] is intentionally one shorter than dates.
        if any(b <= a for a, b in zip(dates, dates[1:], strict=False)):
            raise ValueError(f"{self.ticker}: bars must be strictly date-ascending and unique")
        return self

    def __len__(self) -> int:
        return len(self.bars)

    @property
    def closes(self) -> list[float]:
        """Adjusted close where available, else raw close (one value per bar)."""
        return [b.adj_close if b.adj_close is not None else b.close for b in self.bars]

    @property
    def dates(self) -> list[Date]:
        """Bar dates in ascending order."""
        return [b.date for b in self.bars]


class Fundamentals(BaseModel):
    """Company fundamentals — all metrics optional since coverage varies by provider.

    Intentionally sparse and provider-agnostic: only fields we actually surface in
    the report live here. Provider-specific extras go in ``extra``.
    """

    ticker: str
    as_of: Date | None = None
    revenue: float | None = None
    revenue_growth_yoy: float | None = None  # fractional, e.g. 0.18 = +18%
    earnings: float | None = None
    eps: float | None = None
    pe_ratio: float | None = None
    market_cap: float | None = None
    profit_margin: float | None = None  # fractional
    extra: dict[str, float | str | None] = Field(default_factory=dict)
