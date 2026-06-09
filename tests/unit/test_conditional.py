"""Conditional forward-return study — leakage-safe mechanics (#7), deterministic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_agent.analysis.conditional import conditional_forward_returns


def _series() -> tuple[pd.Series, pd.Series]:
    """Designed case: the driver steps +10% at i=10,20,30 (events at window=1);
    the target steps +5% one bar later, so the event-day forward return is +5%."""
    dates = pd.bdate_range("2020-01-01", periods=40)
    driver = np.full(40, 100.0)
    for e in (10, 20, 30):
        driver[e:] *= 1.10
    target = np.full(40, 100.0)
    for e in (11, 21, 31):
        target[e:] *= 1.05
    return pd.Series(target, index=dates), pd.Series(driver, index=dates)


def test_conditional_isolates_event_forward_returns() -> None:
    t, d = _series()
    study = conditional_forward_returns(
        t, d, target="TGT", driver="DRV", shock_pct=0.05, event_window=1, horizon=1
    )
    assert study.n_events == 3
    assert study.conditional is not None
    assert study.conditional.n == 3
    assert study.conditional.mean == pytest.approx(0.05)  # event-day forward return
    assert study.conditional.prob_up == 1.0
    assert study.baseline.n == 38  # domain = [window, n-horizon)
    assert study.lift_mean is not None and study.lift_mean > 0  # events beat baseline
    assert study.effective_independent_events == 3  # n_events // horizon
    assert study.low_confidence is True  # < 5 effective events


def test_no_leakage_forward_window_strictly_after_event() -> None:
    # Outcome at the last admissible index uses t[i+horizon]; the domain caps at
    # n-horizon so this never reads past the series (no look-ahead / IndexError).
    t, d = _series()
    study = conditional_forward_returns(
        t, d, target="TGT", driver="DRV", shock_pct=0.05, event_window=5, horizon=5
    )
    assert study.n_total == 40 - 5 - 5  # lo=5, hi=35 -> 30 observations
    assert study.horizon_days == 5


def test_direction_down_finds_no_events_here() -> None:
    t, d = _series()  # all driver shocks are positive
    study = conditional_forward_returns(
        t, d, target="TGT", driver="DRV", shock_pct=0.05, event_window=1, horizon=1,
        direction="down",
    )
    assert study.n_events == 0
    assert study.conditional is None
    assert study.lift_mean is None  # nothing to compare


def test_bad_params_raise() -> None:
    t, d = _series()
    with pytest.raises(ValueError):
        conditional_forward_returns(
            t, d, target="T", driver="D", shock_pct=0.0, event_window=1, horizon=1
        )
    with pytest.raises(ValueError):
        conditional_forward_returns(
            t, d, target="T", driver="D", shock_pct=0.05, event_window=0, horizon=1
        )


def test_insufficient_history_raises() -> None:
    dates = pd.bdate_range("2020-01-01", periods=8)
    s = pd.Series(np.linspace(100, 110, 8), index=dates)
    with pytest.raises(ValueError, match="insufficient"):
        conditional_forward_returns(
            s, s, target="T", driver="D", shock_pct=0.05, event_window=5, horizon=5
        )
