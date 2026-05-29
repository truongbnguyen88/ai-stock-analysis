"""Historical-simulation baseline forecaster.

Estimates the forward-return distribution empirically from the stock's own
history: it samples every overlapping ``horizon``-day simple return and reads the
scenario probabilities, expected return, VaR, and a predictive interval directly
off that sample. No distributional assumptions — a strong, hard-to-beat baseline
and the reference against which Phase 5 models are judged.

Caveat: overlapping windows induce autocorrelation (effective sample size < count),
so this understates tail uncertainty somewhat; acceptable for a baseline.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date

import numpy as np

from stock_agent.forecasting.buckets import bucket_probabilities, make_prob_buckets
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceSeries

_MODEL_NAME = "historical_sim"
_MIN_SAMPLES = 30  # below this, flag low confidence


def _horizon_returns(closes: np.ndarray, horizon: int) -> np.ndarray:
    """Overlapping ``horizon``-day simple returns P_{t+h}/P_t - 1."""
    if horizon < 1 or len(closes) <= horizon:
        return np.empty(0, dtype=float)
    return np.asarray(closes[horizon:] / closes[:-horizon] - 1.0, dtype=float)


def historical_forecast(
    closes: Sequence[float],
    horizon_days: int,
    *,
    ticker: str,
    as_of: Date,
    min_samples: int = _MIN_SAMPLES,
) -> ScenarioForecast:
    """Build a ``ScenarioForecast`` from the empirical horizon-return distribution."""
    arr = np.asarray(closes, dtype=float)
    sample = _horizon_returns(arr, horizon_days)
    n = len(sample)
    if n == 0:
        raise ValueError(f"insufficient price history ({len(arr)} bars) for horizon {horizon_days}")

    buckets = make_prob_buckets(bucket_probabilities(sample))
    notes = None
    if n < min_samples:
        notes = f"Only {n} overlapping {horizon_days}-day samples; estimates are low-confidence."

    return ScenarioForecast(
        ticker=ticker,
        as_of=as_of,
        horizon_days=horizon_days,
        model_name=_MODEL_NAME,
        buckets=buckets,
        expected_return=float(sample.mean()),
        upside_prob=float((sample > 0).mean()),
        downside_prob=float((sample < 0).mean()),
        # VaR as a (negative) return at the tail; e.g. var_95 = 5th percentile.
        var_95=float(np.quantile(sample, 0.05)),
        var_99=float(np.quantile(sample, 0.01)),
        # 90% predictive interval on the horizon return.
        ci_level=0.90,
        ci_low=float(np.quantile(sample, 0.05)),
        ci_high=float(np.quantile(sample, 0.95)),
        calibration_status="unknown",  # not yet calibrated (Phase 6)
        notes=notes,
    )


class HistoricalSimulation:
    """``ForecastModel`` using empirical historical horizon returns."""

    name = _MODEL_NAME

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        return historical_forecast(
            series.closes,
            horizon_days,
            ticker=series.ticker,
            as_of=as_of or series.bars[-1].date,
        )
