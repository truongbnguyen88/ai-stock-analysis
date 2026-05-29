"""ML forecaster tests: interface, bucket sums, fallback behaviour."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from stock_agent.forecasting.ml import MLForecaster
from stock_agent.schemas.market import PriceBar, PriceSeries


def _series(n: int, trend: float = 0.001) -> PriceSeries:
    prices = [100.0 * (1 + trend) ** i for i in range(n)]
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i),
            open=p,
            high=p + 0.5,
            low=p - 0.5,
            close=p,
        )
        for i, p in enumerate(prices)
    ]
    return PriceSeries(ticker="X", bars=bars)


@pytest.mark.parametrize("model_type", ["logistic", "xgboost", "lightgbm", "random_forest"])
def test_ml_buckets_sum_to_one(model_type: str) -> None:
    fc = MLForecaster(model_type=model_type, horizon_days=20).forecast(  # type: ignore[arg-type]
        _series(120), horizon_days=20
    )
    total = sum(b.probability for b in fc.buckets)
    assert abs(total - 1.0) < 1e-6


@pytest.mark.parametrize("model_type", ["logistic", "xgboost", "lightgbm", "random_forest"])
def test_ml_probabilities_in_unit_interval(model_type: str) -> None:
    fc = MLForecaster(model_type=model_type, horizon_days=20).forecast(  # type: ignore[arg-type]
        _series(120), horizon_days=20
    )
    for b in fc.buckets:
        assert 0.0 <= b.probability <= 1.0


def test_ml_fallback_to_historical_sim_on_short_series() -> None:
    # Only 20 bars → fewer than min_train_rows=60; should fall back gracefully.
    fc = MLForecaster(model_type="xgboost", horizon_days=10, min_train_rows=60).forecast(
        _series(20), horizon_days=10
    )
    # Fell back → model_name still carries the ML label and notes explains why.
    assert "ml_xgboost" in fc.model_name
    assert fc.notes is not None and "Fell back" in fc.notes


def test_ml_horizon_change_triggers_refit() -> None:
    forecaster = MLForecaster(model_type="xgboost")
    series = _series(120)
    fc1 = forecaster.forecast(series, horizon_days=10)
    fc2 = forecaster.forecast(series, horizon_days=20)  # triggers refit
    assert fc1.horizon_days == 10
    assert fc2.horizon_days == 20
    assert sum(b.probability for b in fc2.buckets) == pytest.approx(1.0, abs=1e-6)
