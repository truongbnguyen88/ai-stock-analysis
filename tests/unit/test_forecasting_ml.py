"""Pooled ML forecaster tests: training, save/load, inference, fallback (offline)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from stock_agent.forecasting.ml import MLForecaster
from stock_agent.forecasting.pooled import PooledModel, train_pooled_from_series
from stock_agent.schemas.market import PriceBar, PriceSeries


def _noisy_series(ticker: str, *, n: int = 260, seed: int = 0) -> PriceSeries:
    """A seeded random-walk series so forward returns span the thresholds."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.02, n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    bars = [
        PriceBar(
            date=date(2022, 1, 1) + timedelta(days=i),
            open=float(p),
            high=float(p) * 1.01,
            low=float(p) * 0.99,
            close=float(p),
        )
        for i, p in enumerate(prices)
    ]
    return PriceSeries(ticker=ticker, bars=bars)


def _universe(n_tickers: int = 6) -> list[PriceSeries]:
    return [_noisy_series(f"T{i}", seed=i) for i in range(n_tickers)]


def test_train_pooled_produces_model() -> None:
    model = train_pooled_from_series(
        _universe(), horizon_days=20, model_type="logistic", min_total_rows=100
    )
    assert model.n_tickers >= 1
    assert model.n_train_rows >= 100
    assert model.imputer is not None  # logistic needs a persisted imputer
    assert model.classifiers  # at least one threshold trained


@pytest.mark.parametrize("model_type", ["logistic", "lightgbm"])
def test_pooled_forecast_buckets_valid(model_type: str) -> None:
    model = train_pooled_from_series(
        _universe(),
        horizon_days=20,
        model_type=model_type,  # type: ignore[arg-type]
        min_total_rows=100,
    )
    fc = MLForecaster(model_type, model=model).forecast(  # type: ignore[arg-type]
        _noisy_series("NVDA", seed=99), horizon_days=20
    )
    assert sum(b.probability for b in fc.buckets) == pytest.approx(1.0, abs=1e-6)
    for b in fc.buckets:
        assert 0.0 <= b.probability <= 1.0
    # Buckets partition at 0, so upside + downside must equal 1 (no double-count
    # of the open tails).
    assert fc.upside_prob + fc.downside_prob == pytest.approx(1.0, abs=1e-6)
    assert fc.model_name == f"ml_{model_type}"
    assert fc.notes is not None and "Pooled" in fc.notes


def test_pooled_save_load_roundtrip(tmp_path: Path) -> None:
    model = train_pooled_from_series(
        _universe(), horizon_days=20, model_type="lightgbm", min_total_rows=100
    )
    path = tmp_path / "pooled_lightgbm_h20.joblib"
    model.save(path)
    assert path.exists()

    loaded = PooledModel.load(path)
    assert loaded.n_train_rows == model.n_train_rows
    assert loaded.feature_cols == model.feature_cols

    fc = MLForecaster("lightgbm", model=loaded).forecast(
        _noisy_series("X", seed=5), horizon_days=20
    )
    assert sum(b.probability for b in fc.buckets) == pytest.approx(1.0, abs=1e-6)


def test_ml_falls_back_to_historical_sim_without_artifact(tmp_path: Path) -> None:
    # Empty models_dir → no artifact → graceful fallback with an explanatory note.
    fc = MLForecaster("lightgbm", models_dir=tmp_path).forecast(
        _noisy_series("X", seed=1), horizon_days=20
    )
    assert "ml_lightgbm" in fc.model_name
    assert fc.notes is not None and "historical_sim" in fc.notes


def test_ml_loads_artifact_from_disk(tmp_path: Path) -> None:
    # Train + persist at the canonical path, then a fresh forecaster loads it.
    model = train_pooled_from_series(
        _universe(), horizon_days=15, model_type="lightgbm", min_total_rows=100
    )
    (tmp_path / "pooled_lightgbm_h15.joblib").parent.mkdir(parents=True, exist_ok=True)
    model.save(tmp_path / "pooled_lightgbm_h15.joblib")

    fc = MLForecaster("lightgbm", models_dir=tmp_path).forecast(
        _noisy_series("NVDA", seed=42), horizon_days=15
    )
    assert fc.notes is not None and "Pooled" in fc.notes  # used the artifact, not fallback
