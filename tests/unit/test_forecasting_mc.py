"""Monte Carlo forecaster tests with known statistical properties."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from stock_agent.forecasting.monte_carlo import MonteCarlo
from stock_agent.schemas.market import PriceBar, PriceSeries


def _flat_series(n: int = 120, base: float = 100.0) -> PriceSeries:
    """Strictly flat series — zero drift, zero vol."""
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i), open=base, high=base, low=base, close=base
        )
        for i in range(n)
    ]
    return PriceSeries(ticker="X", bars=bars)


def _trending_series(n: int = 120, daily_ret: float = 0.001) -> PriceSeries:
    """Constant daily simple return series."""
    prices = [100.0 * (1 + daily_ret) ** i for i in range(n)]
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


def test_gbm_flat_series_expected_return_near_zero() -> None:
    # Zero drift, zero vol → E[r] ≈ 0 for any horizon.
    fc = MonteCarlo(variant="gbm", n_paths=5_000).forecast(_flat_series(), horizon_days=20)
    assert abs(fc.expected_return) < 0.005  # within 0.5%


def test_gbm_reproducible_under_fixed_seed() -> None:
    series = _trending_series()
    fc1 = MonteCarlo(variant="gbm", seed=42).forecast(series, horizon_days=20)
    fc2 = MonteCarlo(variant="gbm", seed=42).forecast(series, horizon_days=20)
    assert fc1.expected_return == fc2.expected_return


def test_gbm_buckets_sum_to_one() -> None:
    fc = MonteCarlo(variant="gbm").forecast(_trending_series(), horizon_days=20)
    assert sum(b.probability for b in fc.buckets) == pytest.approx(1.0, abs=1e-6)


def test_gbm_positive_drift_upside_majority() -> None:
    # Strong uptrend → upside probability should dominate.
    series = _trending_series(daily_ret=0.005)  # +0.5%/day
    fc = MonteCarlo(variant="gbm", n_paths=5_000).forecast(series, horizon_days=20)
    assert fc.upside_prob > 0.6


def test_bootstrap_buckets_sum_to_one() -> None:
    fc = MonteCarlo(variant="bootstrap").forecast(_trending_series(), horizon_days=20)
    assert sum(b.probability for b in fc.buckets) == pytest.approx(1.0, abs=1e-6)


def test_bootstrap_reproducible_under_fixed_seed() -> None:
    series = _trending_series()
    fc1 = MonteCarlo(variant="bootstrap", seed=99).forecast(series, horizon_days=20)
    fc2 = MonteCarlo(variant="bootstrap", seed=99).forecast(series, horizon_days=20)
    assert fc1.expected_return == fc2.expected_return


def test_invalid_variant_raises() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        MonteCarlo(variant="bad")  # type: ignore[arg-type]


def test_insufficient_history_raises() -> None:
    short = PriceSeries(
        ticker="X",
        bars=[
            PriceBar(date=date(2024, 1, i + 1), open=1, high=2, low=0.5, close=1) for i in range(5)
        ],
    )
    with pytest.raises(ValueError, match="insufficient"):
        MonteCarlo().forecast(short, horizon_days=20)


def test_var95_below_expected_return() -> None:
    # VaR (5th percentile) must always be below the expected return.
    fc = MonteCarlo(variant="gbm").forecast(_trending_series(), horizon_days=20)
    assert fc.var_95 is not None
    assert fc.var_95 < fc.expected_return
