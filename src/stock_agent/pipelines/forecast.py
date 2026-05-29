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


# MLForecaster instances are stateful (lazy-trained per series call), so the
# registry returns fresh instances each run to avoid state leaking across tickers.
def _ml(model_type: str, horizon: int) -> MLForecaster:
    return MLForecaster(model_type=model_type, horizon_days=horizon)  # type: ignore[arg-type]


MODEL_REGISTRY: dict[str, ForecastModel] = {
    "historical_sim": HistoricalSimulation(),
    "monte_carlo_gbm": MonteCarlo(variant="gbm"),
    "monte_carlo_bootstrap": MonteCarlo(variant="bootstrap"),
    # ML models instantiated fresh per call (see run_forecast below).
    "logistic": HistoricalSimulation(),  # placeholder; resolved in run_forecast
    "xgboost": HistoricalSimulation(),
    "lightgbm": HistoricalSimulation(),
    "random_forest": HistoricalSimulation(),
}

_ML_MODELS = {"logistic", "xgboost", "lightgbm", "random_forest"}


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
    # ML models are stateful (lazy-trained); instantiate fresh per call so
    # runs on different tickers don't share fitted classifiers.
    if model_name in _ML_MODELS:
        model: ForecastModel = _ml(model_name, horizon_days)
    else:
        model = MODEL_REGISTRY[model_name]

    return model.forecast(series, horizon_days=horizon_days)
