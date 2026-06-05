"""Forecast pipeline: fetch prices and run a forecaster."""

from __future__ import annotations

from stock_agent.data.loader import PriceLoader
from stock_agent.forecasting.base import ForecastModel
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.forecasting.ml import MLForecaster
from stock_agent.forecasting.monte_carlo import MonteCarlo
from stock_agent.forecasting.regime import RegimeForecaster
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.settings import Settings

_PRICE_LOOKBACK_DAYS = 420

_ML_TYPES = ("logistic", "lightgbm")

# Available forecaster names (used for the CLI/agent enum). ML models are built
# per call so they can fetch earnings dates via the registry at inference.
MODEL_NAMES: list[str] = [
    "historical_sim",
    "monte_carlo_gbm",
    "monte_carlo_bootstrap",
    "monte_carlo_jump",
    "monte_carlo_garch",  # GJR-GARCH-t conditional vol (Task 9)
    *_ML_TYPES,
    "ensemble",  # linear probability pool over baselines + GARCH + pooled ML
    "regime_hmm",  # experimental (Task 8); CLI/backtest only, not agent/report
]


def _build_model(model_name: str, registry: ProviderRegistry) -> ForecastModel:
    if model_name == "historical_sim":
        return HistoricalSimulation()
    if model_name == "monte_carlo_gbm":
        return MonteCarlo(variant="gbm")
    if model_name == "monte_carlo_bootstrap":
        return MonteCarlo(variant="bootstrap")
    if model_name == "monte_carlo_jump":
        # Needs the registry to fetch earnings dates for the jump.
        return MonteCarlo(variant="jump", registry=registry)
    if model_name == "monte_carlo_garch":
        return MonteCarlo(variant="garch")
    if model_name in _ML_TYPES:
        # Pass the registry so the earnings feature can be computed at inference.
        return MLForecaster(model_name, registry=registry)  # type: ignore[arg-type]
    if model_name == "ensemble":
        from stock_agent.forecasting.ensemble import full_ensemble

        return full_ensemble(registry)
    if model_name == "regime_hmm":
        return RegimeForecaster()
    raise ValueError(f"unknown model '{model_name}'; available: {MODEL_NAMES}")


def run_forecast(
    ticker: str,
    horizon_days: int,
    *,
    model_name: str = "historical_sim",
    settings: Settings,
    registry: ProviderRegistry | None = None,
) -> ScenarioForecast:
    """Load prices and run the requested forecaster."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unknown model '{model_name}'; available: {MODEL_NAMES}")

    registry = registry or build_default_registry(settings)
    series = (
        PriceLoader(registry).load_recent(ticker.upper(), _PRICE_LOOKBACK_DAYS, min_bars=30).series
    )
    return _build_model(model_name, registry).forecast(series, horizon_days=horizon_days)
