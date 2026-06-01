"""Tests for return buckets and the historical-simulation baseline.

Hand-derived case (horizon=1): closes [100, 110, 121, 108.9] give simple daily
returns [+0.10, +0.10, -0.10]:
  buckets -> "-10% to -5%": 1/3, "> +10%": 2/3
  E[r] = 0.0333..., upside 2/3, downside 1/3
  VaR95 = quantile(0.05) of {-0.1,0.1,0.1} = -0.08 (linear interp)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from stock_agent.forecasting.buckets import (
    DEFAULT_BUCKETS,
    assign_bucket,
    bucket_probabilities,
)
from stock_agent.forecasting.historical import historical_forecast


def test_assign_bucket_boundaries() -> None:
    assert DEFAULT_BUCKETS[assign_bucket(-0.20)][0] == "< -10%"
    assert DEFAULT_BUCKETS[assign_bucket(-0.10)][0] == "-10% to -5%"  # lower inclusive
    assert DEFAULT_BUCKETS[assign_bucket(0.0)][0] == "0% to +5%"
    assert DEFAULT_BUCKETS[assign_bucket(0.10)][0] == "> +10%"  # upper exclusive below


def test_bucket_probabilities_sum_to_one() -> None:
    probs = bucket_probabilities(np.array([0.10, 0.10, -0.10]))
    assert sum(probs) == pytest.approx(1.0)
    assert probs[DEFAULT_BUCKETS.index(("> +10%", 0.10, None))] == pytest.approx(2 / 3)


def test_buckets_scale_with_horizon() -> None:
    from stock_agent.forecasting.buckets import buckets_for_horizon, thresholds_for_horizon

    assert thresholds_for_horizon(20) == [-0.10, -0.05, 0.0, 0.05, 0.10]
    assert thresholds_for_horizon(30) == [-0.20, -0.10, 0.0, 0.10, 0.20]
    assert thresholds_for_horizon(60) == [-0.30, -0.15, 0.0, 0.15, 0.30]
    # labels reflect the scaled band
    assert buckets_for_horizon(60)[-1][0] == "> +30%"
    assert buckets_for_horizon(30)[0][0] == "< -20%"
    # an unlisted horizon snaps to the nearest configured one
    assert thresholds_for_horizon(50) == thresholds_for_horizon(60)


def test_historical_forecast_known_values() -> None:
    fc = historical_forecast(
        [100.0, 110.0, 121.0, 108.9], horizon_days=1, ticker="X", as_of=date(2025, 1, 4)
    )
    assert fc.expected_return == pytest.approx((0.10 + 0.10 - 0.10) / 3)
    assert fc.upside_prob == pytest.approx(2 / 3)
    assert fc.downside_prob == pytest.approx(1 / 3)
    assert fc.var_95 == pytest.approx(-0.08)
    assert sum(b.probability for b in fc.buckets) == pytest.approx(1.0)
    assert fc.calibration_status == "unknown"
    assert fc.notes is not None  # small sample -> low-confidence note


def test_historical_forecast_insufficient_history_raises() -> None:
    with pytest.raises(ValueError):
        historical_forecast([100.0, 101.0], horizon_days=5, ticker="X", as_of=date(2025, 1, 2))
