"""Probabilistic forecast domain models.

CRITICAL INVARIANT: every value here is produced by a statistical or ML model
(see ``forecasting/``), never by the LLM. The LLM may *explain* a
``ScenarioForecast`` but must never construct one.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

# Whether the forecast's probabilities have been validated against realized
# outcomes (see backtesting/calibration.py). Surfaced to users so they know how
# much to trust the numbers.
CalibrationStatus = Literal["unknown", "uncalibrated", "calibrated"]


class ProbBucket(BaseModel):
    """Probability mass for one forward-return range.

    Bounds are fractional returns; ``None`` denotes an open end
    (lower=None => "< upper"; upper=None => "> lower").
    """

    label: str  # human label, e.g. "+5% to +10%"
    lower: float | None  # inclusive lower bound (fractional return)
    upper: float | None  # exclusive upper bound (fractional return)
    probability: float = Field(ge=0.0, le=1.0)


class ScenarioForecast(BaseModel):
    """Model-generated forward-return distribution for one ticker and horizon."""

    ticker: str
    as_of: Date
    horizon_days: int = Field(gt=0)  # trading days
    model_name: str  # e.g. "historical_sim", "monte_carlo_gbm", "xgboost"
    buckets: list[ProbBucket]
    expected_return: float  # fractional

    upside_prob: float = Field(ge=0.0, le=1.0)  # P(return > 0)
    downside_prob: float = Field(ge=0.0, le=1.0)  # P(return < 0)

    # VaR as a (typically negative) fractional return at a confidence level, e.g.
    # var_95 = -0.08 => 5% chance of losing more than 8% over the horizon.
    var_95: float | None = None
    var_99: float | None = None

    # Predictive interval on the return at ``ci_level`` (e.g. 0.90 => 90% CI).
    ci_level: float | None = Field(default=None, ge=0.0, le=1.0)
    ci_low: float | None = None
    ci_high: float | None = None

    calibration_status: CalibrationStatus = "unknown"
    notes: str | None = None  # model caveats, e.g. sparse-data baseline fallback

    @model_validator(mode="after")
    def _check_buckets_sum(self) -> Self:
        # Buckets should partition the outcome space (probabilities sum to ~1).
        # Tolerance absorbs float noise and Monte-Carlo discretization error.
        if self.buckets:
            total = sum(b.probability for b in self.buckets)
            if abs(total - 1.0) > 1e-3:
                raise ValueError(f"bucket probabilities sum to {total:.4f}, expected 1.0")
        return self
