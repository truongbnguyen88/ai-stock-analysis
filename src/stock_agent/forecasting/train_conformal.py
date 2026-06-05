"""Orchestration: compute the pooled conformal interval-corrections offline.

For each (model, horizon): build the model **as of** ``cal_cutoff`` (ML trained on
``<= cal_cutoff``; the ensemble uses those cutoff-trained ML members), forecast every
basket ticker point-in-time across the held-out ``(cal_cutoff, end-h]`` window, pool the
``(CI, realized)`` triples across tickers, and fit one conformal ``q``. Network-bound and
slow (one ML train per model+horizon + lots of forecasts), but a one-time offline step —
the resulting ``conformal.json`` is loaded cheaply at inference.

See ``forecasting/conformal_calibrate.py`` for the leakage argument.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta
from pathlib import Path

from stock_agent.data.market_context import fetch_vix
from stock_agent.forecasting.base import ForecastModel
from stock_agent.forecasting.conformal_calibrate import (
    CONFORMAL_FILE,
    ConformalArtifact,
    fit_entry,
    interval_scores,
    new_artifact,
)
from stock_agent.forecasting.ensemble import EnsembleForecast
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.forecasting.ml import MLForecaster
from stock_agent.forecasting.monte_carlo import MonteCarlo
from stock_agent.forecasting.pooled import train_pooled_from_series
from stock_agent.forecasting.train_pooled import (
    fetch_universe_earnings,
    fetch_universe_series,
    load_universe,
)
from stock_agent.logging_config import get_logger
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.market import PriceSeries
from stock_agent.settings import Settings

log = get_logger(__name__)

_STATELESS = ("historical_sim", "monte_carlo_bootstrap", "monte_carlo_garch")
_ML = ("logistic", "lightgbm")
DEFAULT_MODELS = (*_STATELESS, *_ML, "ensemble")
DEFAULT_HORIZONS = (20, 30, 60)


def _slice(universe: list[PriceSeries], cutoff: Date) -> list[PriceSeries]:
    out = [
        PriceSeries(ticker=s.ticker, bars=[b for b in s.bars if b.date <= cutoff])
        for s in universe
    ]
    return [s for s in out if len(s) >= 60]


def _build_forecaster(
    name: str, ml_at: dict[str, MLForecaster], mc_paths: int
) -> ForecastModel:
    """The forecaster for ``name`` using the cutoff-vintage ML members."""
    if name == "historical_sim":
        return HistoricalSimulation()
    if name == "monte_carlo_bootstrap":
        return MonteCarlo(variant="bootstrap", n_paths=mc_paths)
    if name == "monte_carlo_garch":
        return MonteCarlo(variant="garch", n_paths=mc_paths)
    if name in _ML:
        return ml_at[name]
    return EnsembleForecast(
        [
            HistoricalSimulation(),
            MonteCarlo(variant="bootstrap", n_paths=mc_paths),
            MonteCarlo(variant="garch", n_paths=mc_paths),
            ml_at["logistic"],
            ml_at["lightgbm"],
        ]
    )


def calibrate(
    registry: ProviderRegistry,
    settings: Settings,
    *,
    universe_path: Path = Path("configs/universe.txt"),
    models: tuple[str, ...] = DEFAULT_MODELS,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    cal_cutoff: Date | None = None,
    ci_level: float = 0.90,
    basket_size: int = 24,
    mc_paths: int = 1500,
) -> ConformalArtifact:
    """Compute pooled conformal ``q`` per (model, horizon) and return the artifact."""
    universe = fetch_universe_series(load_universe(universe_path), registry)
    earnings = fetch_universe_earnings([s.ticker for s in universe], registry)
    span = [d for s in universe for d in (s.dates[0], s.dates[-1])]
    vix_full = fetch_vix(registry, start=min(span), end=max(span)) if span else None
    vix = vix_full if (vix_full is not None and not vix_full.empty) else None
    end = max(d for s in universe for d in (s.dates[-1],))
    cutoff = cal_cutoff or (end - timedelta(days=365))
    basket = universe[:basket_size]  # forecasting basket (pooled q is universe-agnostic)

    art = new_artifact(ci_level, cutoff)
    for h in horizons:
        # Train the cutoff-vintage ML once; reuse for the standalone ML + the ensemble.
        sliced = _slice(universe, cutoff)
        ml_at: dict[str, MLForecaster] = {}
        for mt in _ML:
            if mt in models or "ensemble" in models:
                pooled = train_pooled_from_series(
                    sliced, horizon_days=h, model_type=mt,  # type: ignore[arg-type]
                    earnings_by_ticker=earnings, vix=vix, calibrate=settings.calibrate_ml,
                )
                ml_at[mt] = MLForecaster(mt, model=pooled, registry=registry)  # type: ignore[arg-type]

        stride = max(1, h // 2)  # subsample as-ofs to bound cost
        for name in models:
            mdl = _build_forecaster(name, ml_at, mc_paths)
            triples: list[tuple[float, float, float]] = []
            for s in basket:
                triples += interval_scores(mdl, s, h, cal_cutoff=cutoff, stride=stride)
            entry = fit_entry(triples, ci_level=ci_level)
            if entry is None:
                log.warning("conformal.no_data", model=name, horizon=h)
                continue
            # Key by the forecaster's own .name (e.g. "ml_logistic") so inference lookup
            # by forecast.model_name matches.
            art.entries.setdefault(mdl.name, {})[h] = entry
            log.info(
                "conformal.calibrated", model=mdl.name, horizon=h, q=round(entry.q, 4),
                n=entry.n, cov_before=round(entry.coverage_before, 3),
                cov_after=round(entry.coverage_after, 3),
            )
    return art


def conformal_path(settings: Settings) -> Path:
    return Path(settings.output_dir) / "models" / CONFORMAL_FILE
