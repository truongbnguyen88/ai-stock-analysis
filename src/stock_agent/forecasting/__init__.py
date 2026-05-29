"""Probabilistic forecasting — model-generated scenario distributions.

All models implement the same ``ForecastModel`` interface so they are
swappable in pipelines, reports, and backtests.
"""

from stock_agent.forecasting.base import ForecastModel
from stock_agent.forecasting.buckets import (
    DEFAULT_BUCKETS,
    assign_bucket,
    bucket_probabilities,
    make_prob_buckets,
)
from stock_agent.forecasting.historical import (
    HistoricalSimulation,
    historical_forecast,
    sample_to_forecast,
)
from stock_agent.forecasting.ml import MLForecaster
from stock_agent.forecasting.monte_carlo import MonteCarlo

__all__ = [
    "ForecastModel",
    "DEFAULT_BUCKETS",
    "assign_bucket",
    "bucket_probabilities",
    "make_prob_buckets",
    "sample_to_forecast",
    "HistoricalSimulation",
    "historical_forecast",
    "MonteCarlo",
    "MLForecaster",
]
