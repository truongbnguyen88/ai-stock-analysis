"""Monte Carlo forecasting: GBM and block-bootstrap price path simulation.

Both variants simulate N forward paths for a given horizon and read the
scenario distribution off the simulated terminal returns. They share the
same ``ScenarioForecast`` output contract as ``HistoricalSimulation`` so
models are directly comparable in reports and backtests.

GBM (parametric):
  Assumes log-normally distributed daily returns. Estimates drift μ and
  volatility σ from the most recent ``vol_window`` log returns, then
  simulates N independent paths of ``horizon`` daily steps.

Block bootstrap (non-parametric):
  Resamples contiguous blocks of actual log returns with replacement,
  concatenates them to length ``horizon``, and repeats N times. Preserves
  the historical fat tails and short-run autocorrelation that GBM ignores.
  Preferred when the return distribution is clearly non-normal.
"""

from __future__ import annotations

import math
from datetime import date as Date
from typing import Literal

import numpy as np

from stock_agent.forecasting.historical import sample_to_forecast
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceSeries

# Simulation defaults.
_N_PATHS = 10_000
_VOL_WINDOW = 60  # recent days used to estimate μ and σ
_BLOCK_SIZE = 10  # block bootstrap block length (≈ √252 ≈ 16; 10 is typical)


class MonteCarlo:
    """``ForecastModel`` using Monte Carlo path simulation.

    ``variant="gbm"``       — parametric, fast, assumes log-normality.
    ``variant="bootstrap"`` — non-parametric, preserves fat tails.
    """

    def __init__(
        self,
        variant: Literal["gbm", "bootstrap"] = "gbm",
        n_paths: int = _N_PATHS,
        vol_window: int = _VOL_WINDOW,
        block_size: int = _BLOCK_SIZE,
        seed: int = 42,
    ) -> None:
        if variant not in ("gbm", "bootstrap"):
            raise ValueError(f"unknown variant '{variant}'; expected 'gbm' or 'bootstrap'")
        self.name = f"monte_carlo_{variant}"
        self._variant = variant
        self._n_paths = n_paths
        self._vol_window = vol_window
        self._block_size = block_size
        self._seed = seed

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        closes = np.asarray(series.closes, dtype=float)
        # Log returns: shape (T-1,); used for μ/σ estimation and bootstrap pool.
        log_rets = np.diff(np.log(closes))
        if len(log_rets) < max(self._vol_window, self._block_size):
            raise ValueError(f"insufficient price history ({len(closes)} bars) for Monte Carlo")
        sample = (
            self._gbm(log_rets, horizon_days)
            if self._variant == "gbm"
            else self._bootstrap(log_rets, horizon_days)
        )
        return sample_to_forecast(
            sample,
            ticker=series.ticker,
            as_of=as_of or series.bars[-1].date,
            horizon_days=horizon_days,
            model_name=self.name,
        )

    def _gbm(self, log_rets: np.ndarray, horizon: int) -> np.ndarray:
        """GBM simulation: estimate μ, σ from recent history; simulate N paths.

        Each daily increment: (μ - σ²/2)*dt + σ*√dt*Z, Z ~ N(0,1), dt = 1 day.
        The h-day log return is the sum of ``horizon`` daily increments; convert
        to a simple return via expm1.
        """
        recent = log_rets[-self._vol_window :]
        mu = float(recent.mean())
        sigma = float(recent.std(ddof=1))

        rng = np.random.default_rng(self._seed)
        Z = rng.standard_normal((self._n_paths, horizon))
        # Shape: (n_paths, horizon) → sum across horizon axis → (n_paths,)
        daily = (mu - 0.5 * sigma**2) + sigma * Z
        return np.asarray(np.expm1(daily.sum(axis=1)), dtype=float)

    def _bootstrap(self, log_rets: np.ndarray, horizon: int) -> np.ndarray:
        """Block bootstrap: resample contiguous blocks to preserve autocorrelation.

        Fully vectorized via numpy fancy indexing (no Python loop over paths).
        Block size controls the autocorrelation preservation trade-off:
        larger blocks = more autocorrelation preserved but fewer unique blocks.
        """
        T = len(log_rets)
        n_blocks = math.ceil(horizon / self._block_size)
        total = self._n_paths * n_blocks
        max_start = T - self._block_size

        rng = np.random.default_rng(self._seed)
        starts = rng.integers(0, max_start + 1, size=total)
        offsets = np.arange(self._block_size)
        # Fancy index: shape (total, block_size)
        indices = starts[:, None] + offsets[None, :]
        blocks = log_rets[indices]
        # Reshape → (n_paths, n_blocks * block_size), truncate to horizon
        path_log_rets = blocks.reshape(self._n_paths, -1)[:, :horizon]
        return np.asarray(np.expm1(path_log_rets.sum(axis=1)), dtype=float)
