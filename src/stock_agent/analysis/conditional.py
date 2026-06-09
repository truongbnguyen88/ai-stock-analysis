"""Leakage-safe conditional forward-return study (news -> price bridge, #7).

Pure compute over two close-price series. The question: historically, when the
*driver* moved by at least ``shock_pct`` over a trailing ``event_window`` of
trading days, how did the *target*'s forward return over the next ``horizon``
days distribute, versus the unconditional baseline over the same date domain?

LEAKAGE SAFETY (a correctness requirement here):
- The event at date ``t`` uses only the driver's return over ``[t-window, t]`` —
  data available at ``t``.
- The outcome is the target's return over ``[t, t+horizon]`` — STRICTLY after the
  feature cutoff. Feature and target never share a bar.
- Baseline and conditional are drawn from the SAME index domain so the lift is a
  like-for-like comparison.

This is a historical conditional, NOT a forecast and NOT causal — overlapping
event windows make the events autocorrelated, so significance is judged on the
overlap-adjusted ``effective_independent_events``, not the raw count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_agent.schemas.conditional import (
    ConditionalStudy,
    Direction,
    ReturnDistribution,
)

# Below this many non-overlapping events, treat the lift as low-confidence noise.
_MIN_EFFECTIVE_EVENTS = 5


def _distribution(returns: np.ndarray) -> ReturnDistribution | None:
    """Summary stats for a 1-D return sample (``None`` if empty)."""
    if returns.size == 0:
        return None
    return ReturnDistribution(
        n=int(returns.size),
        mean=float(np.mean(returns)),
        median=float(np.median(returns)),
        std=float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
        p05=float(np.percentile(returns, 5)),
        p25=float(np.percentile(returns, 25)),
        p75=float(np.percentile(returns, 75)),
        p95=float(np.percentile(returns, 95)),
        prob_up=float(np.mean(returns > 0.0)),
    )


def conditional_forward_returns(
    target_closes: pd.Series,
    driver_closes: pd.Series,
    *,
    target: str,
    driver: str,
    shock_pct: float,
    event_window: int,
    horizon: int,
    direction: Direction = "both",
) -> ConditionalStudy:
    """Compute the conditional vs unconditional forward-return study (see module doc).

    ``target_closes`` / ``driver_closes`` are date-indexed close prices (ascending).
    ``shock_pct`` is fractional (0.05 = 5%). Raises ``ValueError`` on bad params or
    when the overlapping history is too short for ``event_window + horizon``.
    """
    if shock_pct <= 0:
        raise ValueError("shock_pct must be > 0")
    if event_window < 1 or horizon < 1:
        raise ValueError("event_window and horizon must be >= 1")

    # Align on the shared trading calendar (inner join, drop gaps).
    df = pd.concat({"t": target_closes, "d": driver_closes}, axis=1).dropna()
    n = len(df)
    t = df["t"].to_numpy(dtype=float)
    d = df["d"].to_numpy(dtype=float)

    # Valid event index domain: need a full trailing window AND a full forward horizon.
    lo, hi = event_window, n - horizon
    if hi <= lo:
        raise ValueError(
            f"insufficient overlapping history ({n} shared bars) for "
            f"event_window={event_window} + horizon={horizon}"
        )
    idx = np.arange(lo, hi)
    driver_trailing = d[idx] / d[idx - event_window] - 1.0  # feature: data <= t
    target_forward = t[idx + horizon] / t[idx] - 1.0  # outcome: strictly after t

    finite = np.isfinite(driver_trailing) & np.isfinite(target_forward)
    driver_trailing = driver_trailing[finite]
    target_forward = target_forward[finite]
    if target_forward.size == 0:
        raise ValueError("no finite return observations in the overlapping history")

    if direction == "up":
        events = driver_trailing >= shock_pct
    elif direction == "down":
        events = driver_trailing <= -shock_pct
    else:
        events = np.abs(driver_trailing) >= shock_pct

    conditional = _distribution(target_forward[events])
    baseline = _distribution(target_forward)
    assert baseline is not None  # target_forward is non-empty

    n_events = int(events.sum())
    # Overlap-adjusted independent count: events within `horizon` share outcome bars.
    effective = n_events // horizon if n_events > 0 else 0
    lift_mean = conditional.mean - baseline.mean if conditional is not None else None
    lift_prob_up = conditional.prob_up - baseline.prob_up if conditional is not None else None

    methodology = (
        f"Historical conditional over {target_forward.size} shared trading days: event = "
        f"driver {driver} {direction}-move >= {shock_pct:.0%} over a {event_window}-day trailing "
        f"window; outcome = {target} return over the next {horizon} days (strictly after the "
        "event, so no look-ahead). Baseline is all days in the same domain. Overlapping event "
        "windows are autocorrelated — judge significance on effective_independent_events, not "
        "n_events. Descriptive, not a forecast or recommendation."
    )

    return ConditionalStudy(
        target=target,
        driver=driver,
        shock_pct=shock_pct,
        event_window_days=event_window,
        horizon_days=horizon,
        direction=direction,
        n_total=int(target_forward.size),
        n_events=n_events,
        conditional=conditional,
        baseline=baseline,
        lift_mean=lift_mean,
        lift_prob_up=lift_prob_up,
        effective_independent_events=effective,
        low_confidence=effective < _MIN_EFFECTIVE_EVENTS,
        methodology=methodology,
    )
