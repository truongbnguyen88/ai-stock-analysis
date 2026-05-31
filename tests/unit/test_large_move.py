"""Large-move breakdown: P(|r|>k) split into up/down tails (pure, golden)."""

from __future__ import annotations

from datetime import date

import pytest

from stock_agent.forecasting.buckets import make_prob_buckets
from stock_agent.forecasting.large_move import large_move_breakdown
from stock_agent.schemas.forecast import ScenarioForecast


def _forecast(probs: list[float]) -> ScenarioForecast:
    # buckets: <-10% | -10..-5 | -5..0 | 0..+5 | +5..+10 | >+10%
    return ScenarioForecast(
        ticker="TST",
        as_of=date(2025, 1, 31),
        horizon_days=20,
        model_name="ml_logistic",
        buckets=make_prob_buckets(probs),
        expected_return=0.0,
        upside_prob=sum(probs[3:]),
        downside_prob=sum(probs[:3]),
    )


def test_breakdown_at_10pct_golden() -> None:
    fc = _forecast([0.1, 0.1, 0.1, 0.3, 0.2, 0.2])
    b = large_move_breakdown(fc, threshold=0.10)
    assert b.prob_big_up == pytest.approx(0.2)  # the > +10% bucket
    assert b.prob_big_down == pytest.approx(0.1)  # the < -10% bucket
    assert b.prob_large_move == pytest.approx(0.3)
    assert b.lean == "up"  # 0.2 >= 1.25 * 0.1
    assert b.threshold == 0.10


def test_breakdown_at_5pct_aggregates_outer_two_each_side() -> None:
    fc = _forecast([0.1, 0.1, 0.1, 0.3, 0.2, 0.2])
    b = large_move_breakdown(fc, threshold=0.05)
    assert b.prob_big_up == pytest.approx(0.4)  # +5..+10 (0.2) + >+10 (0.2)
    assert b.prob_big_down == pytest.approx(0.2)  # <-10 (0.1) + -10..-5 (0.1)
    assert b.prob_large_move == pytest.approx(0.6)
    assert b.lean == "up"


def test_balanced_lean() -> None:
    fc = _forecast([0.15, 0.1, 0.25, 0.25, 0.1, 0.15])
    b = large_move_breakdown(fc, threshold=0.10)
    assert b.prob_big_up == pytest.approx(0.15)
    assert b.prob_big_down == pytest.approx(0.15)
    assert b.lean == "balanced"


def test_down_lean() -> None:
    fc = _forecast([0.3, 0.1, 0.1, 0.2, 0.1, 0.2])
    b = large_move_breakdown(fc, threshold=0.10)
    assert b.lean == "down"  # 0.3 >= 1.25 * 0.2


def test_invalid_threshold_raises() -> None:
    fc = _forecast([0.1, 0.1, 0.1, 0.3, 0.2, 0.2])
    with pytest.raises(ValueError, match="threshold must be"):
        large_move_breakdown(fc, threshold=0.07)
