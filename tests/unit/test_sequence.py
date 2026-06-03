"""LSTM sequence forecaster (Task 8 spike): train/forecast, determinism, fallback.

Skipped entirely when the optional ``[sequence]`` extra (torch) is absent — so the
core CI gate, which does not install torch, does not run or require these.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

pytest.importorskip("torch")

from stock_agent.forecasting.base import ForecastModel  # noqa: E402
from stock_agent.forecasting.sequence import (  # noqa: E402
    SequenceForecaster,
    SequenceModel,
    train_sequence_model,
)
from stock_agent.schemas.market import PriceBar, PriceSeries  # noqa: E402


def _series(seed: int, *, n: int = 300, ticker: str | None = None) -> PriceSeries:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    bars = [
        PriceBar(
            date=date(2019, 1, 1) + timedelta(days=i),
            open=float(c),
            high=float(c) * 1.01,
            low=float(c) * 0.99,
            close=float(c),
        )
        for i, c in enumerate(closes)
    ]
    return PriceSeries(ticker=ticker or f"T{seed}", bars=bars)


def _tiny_model() -> SequenceModel:
    universe = [_series(i) for i in range(6)]
    return train_sequence_model(
        universe, horizon=20, lookback=30, hidden=8, layers=1, epochs=2, seed=42
    )


def test_sequence_is_forecastmodel() -> None:
    assert isinstance(SequenceForecaster(_tiny_model()), ForecastModel)


def test_sequence_forecast_valid_and_deterministic() -> None:
    model = _tiny_model()
    target = _series(99, ticker="NVDA")
    fc1 = SequenceForecaster(model).forecast(target, horizon_days=20)
    fc2 = SequenceForecaster(model).forecast(target, horizon_days=20)
    assert fc1.model_name == "lstm_seq"
    assert abs(sum(b.probability for b in fc1.buckets) - 1.0) < 1e-6
    assert fc1 == fc2  # fixed seeds → bit-stable inference


def test_sequence_falls_back_on_short_history() -> None:
    model = _tiny_model()
    short = _series(7, n=25)  # fewer feature rows than lookback (30) → fallback
    fc = SequenceForecaster(model).forecast(short, horizon_days=20)
    assert fc.model_name == "lstm_seq"
    assert fc.notes is not None and "fallback" in fc.notes.lower()
    assert abs(sum(b.probability for b in fc.buckets) - 1.0) < 1e-6


def test_sequence_horizon_mismatch_raises() -> None:
    model = _tiny_model()  # trained at horizon 20
    with pytest.raises(ValueError, match="horizon"):
        SequenceForecaster(model).forecast(_series(1), horizon_days=30)
