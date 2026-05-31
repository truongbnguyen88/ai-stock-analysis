"""Large-move ("big move") reading of a forecast — ML's genuine niche.

A magnitude-first, direction-aware summary derived purely from the bucket
distribution: ``P(|r| > k)`` over the horizon, split into its up/down tails.
Backtesting shows ML predicts these tails with real skill where it cannot predict
direction (see docs/validations_results.md and models_explanation.md §3.6), so
this is the output that surfaces ML's actual value.

Pure function over a ``ScenarioForecast`` — works for any model (baseline or ML),
so the *same* reading is comparable across forecasters.
"""

from __future__ import annotations

from typing import Literal

from stock_agent.schemas.forecast import LargeMoveBreakdown, ScenarioForecast

_TOL = 1e-9
# k must align with a bucket boundary so the tails are exact sums of outer buckets.
_ALLOWED_THRESHOLDS = (0.05, 0.10)


def large_move_breakdown(
    forecast: ScenarioForecast, *, threshold: float = 0.10, lean_ratio: float = 1.25
) -> LargeMoveBreakdown:
    """Summarize a forecast as ``P(|r|>k)`` split into up/down tails.

    Args:
        forecast: any model's bucketed forecast.
        threshold: ``k`` (fractional); must be a bucket boundary (0.05 or 0.10).
        lean_ratio: a tail is the "lean" only if it carries at least this multiple
            of the other tail's mass; otherwise "balanced".

    ``P(r > +k)`` is the mass of buckets at/above ``+k``; ``P(r < -k)`` the mass
    at/below ``-k``; ``P(|r|>k)`` their sum. ``lean`` is a plain-language summary
    of which tail dominates — NOT a directional call on the median.
    """
    if not any(abs(threshold - t) < _TOL for t in _ALLOWED_THRESHOLDS):
        raise ValueError(f"threshold must be one of {_ALLOWED_THRESHOLDS} (got {threshold})")

    up = float(
        sum(
            b.probability
            for b in forecast.buckets
            if b.lower is not None and b.lower >= threshold - _TOL
        )
    )
    down = float(
        sum(
            b.probability
            for b in forecast.buckets
            if b.upper is not None and b.upper <= -threshold + _TOL
        )
    )

    lean: Literal["up", "down", "balanced"]
    if up == 0.0 and down == 0.0:
        lean = "balanced"
    elif up >= lean_ratio * down:
        lean = "up"
    elif down >= lean_ratio * up:
        lean = "down"
    else:
        lean = "balanced"

    return LargeMoveBreakdown(
        ticker=forecast.ticker,
        as_of=forecast.as_of,
        horizon_days=forecast.horizon_days,
        model_name=forecast.model_name,
        threshold=threshold,
        prob_large_move=up + down,
        prob_big_up=up,
        prob_big_down=down,
        lean=lean,
    )
