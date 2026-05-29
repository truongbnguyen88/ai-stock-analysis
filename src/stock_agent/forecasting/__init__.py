"""Probabilistic forecasting — model-generated scenario distributions.

Phase 4 ships the historical-simulation baseline. Monte-Carlo and ML models
(Phase 5) implement the same ``ForecastModel`` interface.
"""

from stock_agent.forecasting.base import ForecastModel
from stock_agent.forecasting.buckets import (
    DEFAULT_BUCKETS,
    assign_bucket,
    bucket_probabilities,
    make_prob_buckets,
)
from stock_agent.forecasting.historical import HistoricalSimulation, historical_forecast

__all__ = [
    "ForecastModel",
    "DEFAULT_BUCKETS",
    "assign_bucket",
    "bucket_probabilities",
    "make_prob_buckets",
    "HistoricalSimulation",
    "historical_forecast",
]
