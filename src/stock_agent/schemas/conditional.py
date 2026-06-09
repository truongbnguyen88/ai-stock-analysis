"""Conditional event-study domain models (news -> price bridge, #7).

A leakage-safe, NON-causal historical conditional: "in the past, when a *driver*
(e.g. an oil ETF) moved by at least X% over a trailing window, how did the
*target* stock's forward return over the next H days distribute, versus its
unconditional baseline?" Every number is computed from price history (see
``analysis.conditional``) — the LLM only maps a news theme to a driver proxy and
narrates; it never produces these figures, and this is not a forecast or advice.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["both", "up", "down"]


class ReturnDistribution(BaseModel):
    """Summary stats of a forward-return sample (fractional returns)."""

    n: int = Field(ge=0)
    mean: float
    median: float
    std: float = Field(ge=0.0)
    p05: float
    p25: float
    p75: float
    p95: float
    prob_up: float = Field(ge=0.0, le=1.0)  # fraction of the sample with return > 0


class ConditionalStudy(BaseModel):
    """Conditional vs unconditional forward-return distribution for a target ticker."""

    target: str
    driver: str
    shock_pct: float = Field(gt=0.0)  # fractional event threshold on the driver move
    event_window_days: int = Field(gt=0)  # trailing window the driver move is measured over
    horizon_days: int = Field(gt=0)  # target forward-return horizon
    direction: Direction

    n_total: int = Field(ge=0)  # comparable sample size (the shared date domain)
    n_events: int = Field(ge=0)  # dates that met the driver-shock condition

    conditional: ReturnDistribution | None  # None when no events were found
    baseline: ReturnDistribution

    # Conditional minus baseline (None when there were no events). The headline.
    lift_mean: float | None
    lift_prob_up: float | None

    # Forward windows of events within ``horizon`` overlap -> autocorrelated; this
    # is the rough count of non-overlapping events (n_events // horizon), a more
    # honest "effective sample size" for judging significance.
    effective_independent_events: int = Field(ge=0)
    low_confidence: bool  # too few (effective) events to read much into the lift
    methodology: str
