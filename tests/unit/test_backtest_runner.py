"""Backtest runner: exceedance mapping, point-in-time slicing, per-fold refit."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from stock_agent.backtesting.runner import (
    exceedance_probabilities,
    run_backtest,
    stateless_builder,
)
from stock_agent.forecasting.buckets import make_prob_buckets
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceBar, PriceSeries

_START = date(2020, 1, 1)
_BUCKET_PROBS = [0.1, 0.1, 0.1, 0.3, 0.2, 0.2]


def _forecast(ticker: str, as_of: date, horizon: int) -> ScenarioForecast:
    return ScenarioForecast(
        ticker=ticker,
        as_of=as_of,
        horizon_days=horizon,
        model_name="recording",
        buckets=make_prob_buckets(_BUCKET_PROBS),
        expected_return=0.0,
        upside_prob=0.7,
        downside_prob=0.3,
    )


def _series(n: int = 600) -> PriceSeries:
    bars = []
    price = 100.0
    for i in range(n):
        price *= 1.001 if i % 2 == 0 else 0.9994
        c = round(price, 4)
        bars.append(
            PriceBar(
                date=_START + timedelta(days=i),
                open=c,
                high=round(c * 1.01, 4),
                low=round(c * 0.99, 4),
                close=c,
            )
        )
    return PriceSeries(ticker="TST", bars=bars)


def test_exceedance_probabilities_golden() -> None:
    fc = _forecast("TST", _START, 20)
    # thresholds [-0.10,-0.05,0,0.05,0.10] → cumulative tail sums of the buckets.
    ex = exceedance_probabilities(fc)
    assert ex == pytest.approx([0.9, 0.8, 0.7, 0.4, 0.2])


class _RecordingModel:
    """Records the as-of (last bar date) of every series it is asked to forecast."""

    name = "recording"

    def __init__(self) -> None:
        self.seen_last_dates: list[date] = []

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: date | None = None
    ) -> ScenarioForecast:
        # The slice's final bar MUST equal as_of — i.e. no future data leaked in.
        assert series.dates[-1] == as_of
        self.seen_last_dates.append(series.dates[-1])
        return _forecast(series.ticker, as_of or series.dates[-1], horizon_days)


def test_runner_is_point_in_time_and_counts_match() -> None:
    series = _series(600)
    model = _RecordingModel()
    result = run_backtest(
        series,
        stateless_builder(model),
        model_name="recording",
        horizon_days=20,
        min_train=252,
        test_size=4,
    )
    # Every forecast saw a slice ending exactly at its as-of (assert inside model).
    assert len(model.seen_last_dates) == result.n_predictions
    assert result.n_predictions > 0
    # Constant bucket forecast → per-threshold metrics + calibration are populated.
    assert len(result.thresholds) == 5
    assert result.calibration.n == result.n_predictions * 5
    assert result.as_of_start < result.as_of_end


def test_runner_refits_per_fold_with_correct_cutoff() -> None:
    series = _series(600)
    seen_cutoffs: list[date] = []

    def builder(train_end: date) -> _RecordingModel:
        seen_cutoffs.append(train_end)
        return _RecordingModel()

    result = run_backtest(
        series,
        builder,
        model_name="recording",
        horizon_days=20,
        min_train=252,
        test_size=4,
    )
    # build_model called once per fold; each cutoff strictly precedes its test block.
    assert len(seen_cutoffs) == result.n_folds
    assert seen_cutoffs == sorted(seen_cutoffs)  # folds advance in time


def test_runner_raises_on_short_series() -> None:
    with pytest.raises(ValueError):
        run_backtest(
            _series(100),
            stateless_builder(_RecordingModel()),
            model_name="recording",
            horizon_days=20,
            min_train=252,
        )
