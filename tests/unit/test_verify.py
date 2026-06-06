"""Artifact verification gate (network-free)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from stock_agent.forecasting.pooled import default_model_path, train_pooled_from_series
from stock_agent.forecasting.verify import verify_artifacts
from stock_agent.schemas.market import PriceBar, PriceSeries


def _series(ticker: str, *, n: int = 260, seed: int = 0) -> PriceSeries:
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    return PriceSeries(
        ticker=ticker,
        bars=[
            PriceBar(
                date=date(2022, 1, 1) + timedelta(days=i),
                open=float(p),
                high=float(p) * 1.01,
                low=float(p) * 0.99,
                close=float(p),
            )
            for i, p in enumerate(prices)
        ],
    )


def _train_and_save(models_dir: Path, model: str, horizon: int) -> None:
    m = train_pooled_from_series(
        [_series(f"T{i}", seed=i) for i in range(6)],
        horizon_days=horizon,
        model_type=model,  # type: ignore[arg-type]
        min_total_rows=100,
        calibrate=False,  # fast; verify is structural, not calibration-specific
    )
    m.save(default_model_path(models_dir, model, horizon))


def test_verify_passes_for_good_artifacts(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _train_and_save(models_dir, "logistic", 20)
    _train_and_save(models_dir, "lightgbm", 20)
    assert verify_artifacts(models_dir, models=["logistic", "lightgbm"], horizons=[20]) == []


def test_verify_flags_missing_artifact(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    _train_and_save(models_dir, "logistic", 20)  # lightgbm intentionally not trained
    problems = verify_artifacts(models_dir, models=["logistic", "lightgbm"], horizons=[20])
    assert any("lightgbm h20" in p and "missing" in p for p in problems)


def test_verify_flags_degraded_data(tmp_path: Path) -> None:
    # A structurally-valid artifact trained on too few tickers/rows (the degraded
    # data-month failure mode) must be rejected by the data-quality floor.
    models_dir = tmp_path / "models"
    _train_and_save(models_dir, "logistic", 20)  # 6 synthetic tickers, small row count
    problems = verify_artifacts(
        models_dir, models=["logistic"], horizons=[20], min_tickers=50, min_rows=100_000
    )
    assert any("tickers" in p and "degraded" in p for p in problems)
    assert any("rows" in p and "degraded" in p for p in problems)
    # Without the floor (defaults 0) the same artifact passes — gate is opt-in.
    assert verify_artifacts(models_dir, models=["logistic"], horizons=[20]) == []
    # Row floor fires independently of the ticker floor.
    rows_only = verify_artifacts(
        models_dir, models=["logistic"], horizons=[20], min_tickers=0, min_rows=100_000
    )
    assert any("rows" in p for p in rows_only)
    assert not any("tickers" in p for p in rows_only)


def _write_conformal(models_dir: Path, *, cov_after: float = 0.90, q: float = 0.05) -> None:
    from stock_agent.forecasting.conformal_calibrate import ConformalEntry, new_artifact

    art = new_artifact(0.90, date(2025, 1, 1))
    for m in ("ensemble", "ml_logistic", "ml_lightgbm"):
        art.entries[m] = {
            h: ConformalEntry(q=q, n=300, coverage_before=0.80, coverage_after=cov_after)
            for h in (20, 30, 60)
        }
    from stock_agent.forecasting.conformal_calibrate import CONFORMAL_FILE

    art.save(models_dir / CONFORMAL_FILE)


def test_verify_conformal_ok(tmp_path: Path) -> None:
    from stock_agent.forecasting.verify import verify_conformal

    _write_conformal(tmp_path)
    problems = verify_conformal(
        tmp_path, required_models=["ensemble", "ml_logistic", "ml_lightgbm"], horizons=[20, 30, 60]
    )
    assert problems == []


def test_verify_conformal_missing_artifact(tmp_path: Path) -> None:
    from stock_agent.forecasting.verify import verify_conformal

    problems = verify_conformal(tmp_path, required_models=["ensemble"], horizons=[20])
    assert problems and "missing" in problems[0]


def test_verify_conformal_flags_undercoverage_and_missing_cell(tmp_path: Path) -> None:
    from stock_agent.forecasting.verify import verify_conformal

    _write_conformal(tmp_path, cov_after=0.78)  # conformal failed to reach target
    problems = verify_conformal(
        tmp_path,
        required_models=["ensemble", "monte_carlo_garch"],  # garch not written → missing
        horizons=[20],
    )
    assert any("coverage 0.78" in p for p in problems)  # under-coverage flagged
    assert any("monte_carlo_garch" in p and "missing" in p for p in problems)
