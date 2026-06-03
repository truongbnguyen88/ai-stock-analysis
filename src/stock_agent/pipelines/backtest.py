"""Backtest pipeline: fetch → walk-forward evaluate → log experiment.

Thin orchestration over ``backtesting/``. Default compares the leakage-safe,
offline stateless forecasters (historical-sim + Monte-Carlo) on identical folds
— the architecture's "baselines as guardrails". ML models are supported via a
per-fold pooled refit (``build_model`` rebuilds the pooled model on universe
data ``<= train_end`` each fold) but are opt-in and heavier (network + training
per fold).

Every run is logged to ``outputs/experiments/<run_id>/`` for reproducibility:
config (params, seed, data window) + each model's serialized result.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path

from stock_agent.backtesting.runner import ModelBuilder, run_backtest, stateless_builder
from stock_agent.data.loader import PriceLoader
from stock_agent.forecasting.base import ForecastModel
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.forecasting.monte_carlo import MonteCarlo
from stock_agent.logging_config import get_logger
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.schemas.backtest import BacktestResult
from stock_agent.schemas.market import PriceSeries
from stock_agent.settings import Settings

log = get_logger(__name__)

# Leakage-safe, offline, no-refit forecasters — the default comparison set.
STATELESS_MODELS: tuple[str, ...] = (
    "historical_sim",
    "monte_carlo_gbm",
    "monte_carlo_bootstrap",
    "monte_carlo_garch",  # validated peer (Task 9): best at h30/h60, ties at h20
)
_ML_MODELS: tuple[str, ...] = ("logistic", "lightgbm")
_BACKTEST_LOOKBACK_DAYS = 2200  # ~6y: warmup + enough OOS folds
_MC_PATHS = 5000  # fewer paths than live (snappier; bucket probs stable enough)


def _stateless_model(name: str, registry: ProviderRegistry) -> ForecastModel:
    if name == "historical_sim":
        return HistoricalSimulation()
    if name == "monte_carlo_gbm":
        return MonteCarlo(variant="gbm", n_paths=_MC_PATHS)
    if name == "monte_carlo_bootstrap":
        return MonteCarlo(variant="bootstrap", n_paths=_MC_PATHS)
    if name == "monte_carlo_jump":
        return MonteCarlo(variant="jump", n_paths=_MC_PATHS, registry=registry)
    if name == "monte_carlo_garch":
        return MonteCarlo(variant="garch", n_paths=_MC_PATHS)
    if name == "regime_hmm":
        # Experimental (Task 8): fit point-in-time inside forecast(), so the
        # runner's per-as-of slicing keeps it leakage-safe like the other
        # stateless models. Opt-in only (not in the default comparison set).
        from stock_agent.forecasting.regime import RegimeForecaster

        return RegimeForecaster()
    raise ValueError(f"'{name}' is not a stateless backtest model")


def _pooled_builder(
    model_type: str,
    horizon_days: int,
    registry: ProviderRegistry,
    universe_path: Path,
    *,
    calibrate: bool = True,
) -> ModelBuilder:
    """Per-fold pooled-ML builder: trains on universe data ``<= train_end``.

    Fetches the universe once; each fold slices every universe series to the
    training cutoff and fits a fresh pooled model, so the backtest never trains
    on data at/after the test fold (leakage-safe pooled walk-forward).
    """
    from stock_agent.data.market_context import fetch_vix
    from stock_agent.forecasting.ml import MLForecaster
    from stock_agent.forecasting.pooled import ModelType, train_pooled_from_series
    from stock_agent.forecasting.train_pooled import (
        fetch_universe_earnings,
        fetch_universe_series,
        load_universe,
    )

    tickers = load_universe(universe_path)
    universe = fetch_universe_series(tickers, registry)
    earnings = fetch_universe_earnings([s.ticker for s in universe], registry)
    mt: ModelType = model_type  # type: ignore[assignment]
    # VIX fetched once over the universe span; build_price_feature_matrix reindexes it
    # to each fold's sliced (<= train_end) dates, so only past VIX is ever used.
    _span = [d for s in universe for d in (s.dates[0], s.dates[-1])]
    _vix = fetch_vix(registry, start=min(_span), end=max(_span)) if _span else None
    vix = _vix if (_vix is not None and not _vix.empty) else None

    def build(train_end: Date) -> ForecastModel:
        sliced = [
            PriceSeries(ticker=s.ticker, bars=[b for b in s.bars if b.date <= train_end])
            for s in universe
        ]
        sliced = [s for s in sliced if len(s) >= 60]  # drop too-short slices
        pooled = train_pooled_from_series(
            sliced, horizon_days=horizon_days, model_type=mt, earnings_by_ticker=earnings,
            vix=vix, calibrate=calibrate,
        )
        return MLForecaster(mt, model=pooled, registry=registry)

    return build


def run_backtest_pipeline(
    ticker: str,
    horizon_days: int,
    *,
    model_names: list[str],
    settings: Settings,
    registry: ProviderRegistry | None = None,
    universe_path: Path = Path("configs/universe.txt"),
    min_train: int = 252,
    test_size: int = 6,
    big_move_k: float = 0.10,
    log_experiment: bool = True,
    calibrate: bool = True,
) -> dict[str, BacktestResult]:
    """Backtest one or more forecasters on a ticker; return {model_name: result}.

    Stateless models run offline and fast; ML model names trigger a per-fold
    pooled refit over the universe (slow). Results are logged unless disabled.
    """
    registry = registry or build_default_registry(settings)
    series = (
        PriceLoader(registry)
        .load_recent(ticker.upper(), _BACKTEST_LOOKBACK_DAYS, min_bars=min_train + 2 * horizon_days)
        .series
    )

    results: dict[str, BacktestResult] = {}
    for name in model_names:
        refit_stateless = ("monte_carlo_jump", "regime_hmm")  # opt-in (not in the default set)
        if name in STATELESS_MODELS or name in refit_stateless:
            builder = stateless_builder(_stateless_model(name, registry))
            model_label = name
        elif name in _ML_MODELS:
            log.warning("backtest.ml_refit_slow", model=name)
            builder = _pooled_builder(
                name, horizon_days, registry, universe_path, calibrate=calibrate
            )
            model_label = f"ml_{name}"
        else:
            raise ValueError(f"unknown backtest model '{name}'")

        results[name] = run_backtest(
            series,
            builder,
            model_name=model_label,
            horizon_days=horizon_days,
            min_train=min_train,
            test_size=test_size,
            seed=settings.random_seed,
            big_move_k=big_move_k,
        )

    if log_experiment:
        _log_experiment(settings, ticker, horizon_days, series, results, min_train, test_size)
    return results


def _log_experiment(
    settings: Settings,
    ticker: str,
    horizon_days: int,
    series: PriceSeries,
    results: dict[str, BacktestResult],
    min_train: int,
    test_size: int,
) -> Path:
    """Persist config + results to ``outputs/experiments/<run_id>/`` (reproducible)."""
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{ticker.upper()}_h{horizon_days}"
    run_dir = Path(settings.output_dir) / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_id,
        "ticker": ticker.upper(),
        "horizon_days": horizon_days,
        "models": list(results.keys()),
        "min_train": min_train,
        "test_size": test_size,
        "embargo": horizon_days,
        "stride": horizon_days,
        "seed": settings.random_seed,
        "n_bars": len(series),
        "data_start": series.dates[0].isoformat(),
        "data_end": series.dates[-1].isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for name, result in results.items():
        (run_dir / f"result_{name}.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8"
        )
    log.info("backtest.logged", run_dir=str(run_dir))
    return run_dir
