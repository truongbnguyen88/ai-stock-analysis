"""Forecast pipeline: fetch prices and run a forecaster."""

from __future__ import annotations

from stock_agent.data.loader import PriceLoader
from stock_agent.forecasting.base import ForecastModel
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.forecasting.ml import MLForecaster
from stock_agent.forecasting.monte_carlo import MonteCarlo
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.settings import Settings

_PRICE_LOOKBACK_DAYS = 420

# Pooled ML forecasters load a shared, ticker-agnostic artifact (cached on first
# use), so one instance is safe to reuse across tickers — no per-ticker state.
MODEL_REGISTRY: dict[str, ForecastModel] = {
    "historical_sim": HistoricalSimulation(),
    "monte_carlo_gbm": MonteCarlo(variant="gbm"),
    "monte_carlo_bootstrap": MonteCarlo(variant="bootstrap"),
    "logistic": MLForecaster("logistic"),
    "xgboost": MLForecaster("xgboost"),
    "lightgbm": MLForecaster("lightgbm"),
    "random_forest": MLForecaster("random_forest"),
}


def run_forecast(
    ticker: str,
    horizon_days: int,
    *,
    model_name: str = "historical_sim",
    settings: Settings,
    registry: ProviderRegistry | None = None,
) -> ScenarioForecast:
    """Load prices and run the requested forecaster."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{model_name}'; available: {list(MODEL_REGISTRY)}")

    registry = registry or build_default_registry(settings)
    series = (
        PriceLoader(registry).load_recent(ticker.upper(), _PRICE_LOOKBACK_DAYS, min_bars=30).series
    )
    return MODEL_REGISTRY[model_name].forecast(series, horizon_days=horizon_days)
