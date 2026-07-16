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
    model_name: str = "ensemble",
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
    forecast = _build_model(model_name, registry).forecast(series, horizon_days=horizon_days)
    return apply_conformal(forecast, settings)


def apply_conformal(forecast: ScenarioForecast, settings: Settings) -> ScenarioForecast:
    """Widen/​tighten the served CI + VaR by the offline pooled conformal ``q`` (if any).

    ``ci_low``/``var_95`` are the same 5% lower quantile, so both shift by ``-q``;
    ``ci_high`` by ``+q``; ``var_99`` (1% level) shifts by ``-q`` too (approximate but
    conservative — the correction is calibrated to the CI level). No-op without an artifact.
    """
    if not settings.conformal_intervals or forecast.ci_low is None or forecast.ci_high is None:
        return forecast
    from stock_agent.forecasting.conformal import conformalize_interval
    from stock_agent.forecasting.conformal_calibrate import ConformalArtifact
    from stock_agent.forecasting.train_conformal import conformal_path

    art = ConformalArtifact.load(conformal_path(settings))
    if art is None:
        return forecast
    q = art.q_for(forecast.model_name, forecast.horizon_days)
    if q is None:
        return forecast
    lo, hi = conformalize_interval(forecast.ci_low, forecast.ci_high, q)
    update: dict[str, object] = {"ci_low": lo, "ci_high": hi}
    if forecast.var_95 is not None:
        update["var_95"] = forecast.var_95 - q
    if forecast.var_99 is not None:
        update["var_99"] = forecast.var_99 - q
    note = f"CI/VaR conformally calibrated (q={q:+.3f} @ {art.ci_level:.0%} target)."
    update["notes"] = f"{forecast.notes} {note}".strip() if forecast.notes else note
    return forecast.model_copy(update=update)
