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

Earnings jump (jump-diffusion):
  GBM, plus a jump on the path when an earnings announcement falls inside the
  horizon. The jump is bootstrapped from the stock's OWN historical post-earnings
  moves (close-before → close-after each past earnings), so it captures real,
  asymmetric event risk and widens/fattens the terminal distribution — exactly
  the binary catalyst a plain price model can't see. All inputs are data
  (earnings dates + historical moves + estimated vol); no figure comes from an LLM.
"""

from __future__ import annotations

import bisect
import math
from datetime import date as Date
from datetime import timedelta
from typing import Literal

import numpy as np

from stock_agent.forecasting.historical import sample_to_forecast
from stock_agent.logging_config import get_logger
from stock_agent.providers.base import ProviderError
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)

# Simulation defaults.
_N_PATHS = 10_000
_VOL_WINDOW = 60  # recent days used to estimate μ and σ
_BLOCK_SIZE = 10  # block bootstrap block length (≈ √252 ≈ 16; 10 is typical)
_GARCH_MIN_OBS = 250  # daily returns needed for a stable GJR-GARCH fit; else bootstrap
_MIN_JUMP_SAMPLES = 8  # below this many past earnings moves, skip the jump
_TRADING_TO_CAL = 365.0 / 252.0  # trading days → calendar days
# Earnings recur ~quarterly, so the ~420d forecast window holds only ~4-5 moves —
# too few to calibrate a jump. Pull ~4y (~16 earnings) just for calibration.
_CALIBRATION_LOOKBACK_DAYS = 1460


def historical_earnings_moves(series: PriceSeries, earnings_dates: list[Date]) -> np.ndarray:
    """Empirical post-earnings log moves: close-before → close-after each PAST date.

    Captures the realized earnings reaction (the announcement is typically after
    close, so the move spans into the next trading day). Future earnings dates
    have no realized move and are skipped. Pure function (offline-testable).
    """
    dates = series.dates
    closes = series.closes  # adjusted close where available
    moves: list[float] = []
    for e in earnings_dates:
        i = bisect.bisect_right(dates, e) - 1  # last bar on/before the announcement
        if i < 0 or i + 1 >= len(dates):
            continue  # no bar after (future earnings) or before
        before, after = closes[i], closes[i + 1]
        if before > 0 and after > 0:
            moves.append(math.log(after / before))
    return np.asarray(moves, dtype=float)


class MonteCarlo:
    """``ForecastModel`` using Monte Carlo path simulation.

    ``variant="gbm"``       — parametric, fast, assumes log-normality.
    ``variant="bootstrap"`` — non-parametric, preserves fat tails.
    ``variant="jump"``      — GBM + an earnings jump when earnings fall in the
                              horizon (needs ``registry`` for earnings dates).
    ``variant="garch"``     — GJR-GARCH(1,1)-t: forecasts conditional volatility
                              FORWARD (mean-reverting) with the equity leverage
                              effect + fat tails. Unlike GBM (constant vol) and the
                              bootstrap (recent-vol resampling), it models how vol
                              evolves over the horizon — sharper tails/VaR when
                              current vol differs from the long-run level.
    """

    def __init__(
        self,
        variant: Literal["gbm", "bootstrap", "jump", "garch"] = "gbm",
        n_paths: int = _N_PATHS,
        vol_window: int = _VOL_WINDOW,
        block_size: int = _BLOCK_SIZE,
        seed: int = 42,
        registry: ProviderRegistry | None = None,
    ) -> None:
        if variant not in ("gbm", "bootstrap", "jump", "garch"):
            raise ValueError(f"unknown variant '{variant}'; expected gbm/bootstrap/jump/garch")
        self.name = f"monte_carlo_{variant}"
        self._variant = variant
        self._n_paths = n_paths
        self._vol_window = vol_window
        self._block_size = block_size
        self._seed = seed
        self._registry = registry  # needed only for the jump variant

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        as_of = as_of or series.bars[-1].date
        closes = np.asarray(series.closes, dtype=float)
        # Log returns: shape (T-1,); used for μ/σ estimation and bootstrap pool.
        log_rets = np.diff(np.log(closes))
        if len(log_rets) < max(self._vol_window, self._block_size):
            raise ValueError(f"insufficient price history ({len(closes)} bars) for Monte Carlo")

        note: str | None = None
        if self._variant == "gbm":
            sample = self._gbm(log_rets, horizon_days)
        elif self._variant == "bootstrap":
            sample = self._bootstrap(log_rets, horizon_days)
        elif self._variant == "garch":
            sample, note = self._garch(series.ticker, log_rets, horizon_days)
        else:
            sample, note = self._jump(series, log_rets, horizon_days, as_of)

        fc = sample_to_forecast(
            sample,
            ticker=series.ticker,
            as_of=as_of,
            horizon_days=horizon_days,
            model_name=self.name,
        )
        return fc.model_copy(update={"notes": note}) if note else fc

    def _gbm_terminal_log(self, log_rets: np.ndarray, horizon: int) -> np.ndarray:
        """Terminal h-day LOG returns under GBM (sum of daily increments)."""
        recent = log_rets[-self._vol_window :]
        mu = float(recent.mean())
        sigma = float(recent.std(ddof=1))
        rng = np.random.default_rng(self._seed)
        Z = rng.standard_normal((self._n_paths, horizon))
        daily = (mu - 0.5 * sigma**2) + sigma * Z
        return np.asarray(daily.sum(axis=1), dtype=float)

    def _gbm(self, log_rets: np.ndarray, horizon: int) -> np.ndarray:
        """GBM terminal SIMPLE returns (expm1 of the terminal log returns)."""
        return np.asarray(np.expm1(self._gbm_terminal_log(log_rets, horizon)), dtype=float)

    def _garch(
        self, ticker: str, log_rets: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, str | None]:
        """GJR-GARCH(1,1)-t forward simulation of terminal h-day SIMPLE returns.

        Fits a Glosten-Jagannathan-Runkle GARCH (the ``o=1`` term captures the
        equity *leverage effect* — volatility rises more after negative shocks)
        with Student-t innovations (fat tails), then simulates ``n_paths`` forward
        paths from the *current* conditional variance. Because GARCH variance
        forecasts mean-revert toward the long-run level, the terminal distribution
        widens/narrows correctly when today's vol is unusually high/low — the
        information GBM (constant vol) and the bootstrap (recent-vol resampling)
        lack at multi-day horizons.

        Degrades safely: insufficient history, a missing ``arch`` install, or any
        fit/simulation failure falls back to the block bootstrap (with a note).
        Deterministic — a seeded ``RandomState`` drives the simulation and the MLE
        starts from fixed values.
        """
        if len(log_rets) < _GARCH_MIN_OBS:
            note = f"{len(log_rets)} returns < {_GARCH_MIN_OBS} min for GARCH; bootstrap fallback."
            return self._bootstrap(log_rets, horizon), note
        try:
            import warnings

            from arch.univariate import GARCH, ConstantMean, StudentsT

            r = 100.0 * log_rets  # scale to ~O(1) so the MLE is well-conditioned
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # convergence/scale warnings → handled by fallback
                # Explicit construction (arch_model only takes a string dist): a
                # seeded Student-t distribution is what makes the simulation
                # reproducible — forecast(random_state=) is ignored by arch.
                am = ConstantMean(r)
                am.volatility = GARCH(p=1, o=1, q=1)  # o=1 → GJR leverage term
                am.distribution = StudentsT(seed=self._seed)
                res = am.fit(disp="off")
                sim = res.forecast(
                    horizon=horizon,
                    method="simulation",
                    simulations=self._n_paths,
                    reindex=False,
                )
            values = sim.simulations.values
            if values is None:
                raise ValueError("GARCH simulation returned no values")
            paths = values[0]  # (n_paths, horizon), percent log returns
            sample = np.asarray(np.expm1(paths.sum(axis=1) / 100.0), dtype=float)
            if not np.all(np.isfinite(sample)):
                raise ValueError("non-finite GARCH simulation")
            return sample, None
        except Exception as exc:  # noqa: BLE001 - any fit/sim failure → safe bootstrap fallback
            log.info("garch.fallback", ticker=ticker, error=str(exc))
            note = f"GARCH fit failed ({exc}); bootstrap fallback."
            return self._bootstrap(log_rets, horizon), note

    def _earnings_dates(self, ticker: str) -> list[Date] | None:
        if self._registry is None:
            return None
        try:
            return self._registry.get_earnings_dates(ticker)
        except ProviderError:
            return None

    def _calibration_series(self, series: PriceSeries, as_of: Date) -> PriceSeries:
        """Longer price history for calibrating the earnings jump.

        The forecast ``series`` (~420d) holds only ~4-5 quarterly earnings — too
        few to characterize the jump. Pull ~4y so ~16 historical moves are
        available. Point-in-time preserved: the fetch ends at ``as_of``. Falls back
        to the passed ``series`` when no registry / the extended fetch is
        unavailable (the jump then uses whatever moves the window contains, and the
        ``_MIN_JUMP_SAMPLES`` gate may legitimately fall back to GBM).
        """
        if self._registry is None:
            return series
        try:
            start = as_of - timedelta(days=_CALIBRATION_LOOKBACK_DAYS)
            return self._registry.get_daily_prices(series.ticker, start, as_of)
        except ProviderError:
            return series

    def _jump(
        self, series: PriceSeries, log_rets: np.ndarray, horizon: int, as_of: Date
    ) -> tuple[np.ndarray, str | None]:
        """GBM terminal returns plus an earnings jump if earnings fall in the horizon.

        Adds one bootstrapped historical post-earnings move to each path's terminal
        log return when the next earnings date lands within the horizon. Falls back
        to plain GBM (with an explanatory note) when there's no earnings data, too
        few historical moves, or earnings are beyond the horizon.
        """
        terminal_log = self._gbm_terminal_log(log_rets, horizon)

        earnings = self._earnings_dates(series.ticker)
        if not earnings:
            return np.asarray(np.expm1(terminal_log), dtype=float), "No earnings data; GBM only."

        next_e = next((e for e in sorted(earnings) if e > as_of), None)
        if next_e is None:
            return np.asarray(
                np.expm1(terminal_log), dtype=float
            ), "No upcoming earnings; GBM only."

        # Trading-day horizon → calendar days for the date comparison.
        if (next_e - as_of).days > horizon * _TRADING_TO_CAL:
            return (
                np.asarray(np.expm1(terminal_log), dtype=float),
                f"Next earnings {next_e} beyond the {horizon}d horizon; GBM only.",
            )

        moves = historical_earnings_moves(self._calibration_series(series, as_of), earnings)
        if len(moves) < _MIN_JUMP_SAMPLES:
            return (
                np.asarray(np.expm1(terminal_log), dtype=float),
                f"Only {len(moves)} past earnings moves (<{_MIN_JUMP_SAMPLES}); GBM only.",
            )

        # Replace the earnings day with the empirical jump: re-simulate horizon-1
        # "normal" GBM days and add one bootstrapped post-earnings move, so the
        # earnings day isn't counted as both a diffusion day and a jump day. The
        # moves are used as-is (mean-inclusive), so the path keeps the stock's real
        # post-earnings skew rather than imposing a zero-mean shock.
        rng = np.random.default_rng(self._seed + 1)
        terminal_log = self._gbm_terminal_log(log_rets, horizon - 1) + rng.choice(
            moves, size=self._n_paths
        )
        note = (
            f"Earnings jump applied: {next_e} in horizon, "
            f"{len(moves)} historical moves (σ={moves.std(ddof=1):.1%})."
        )
        return np.asarray(np.expm1(terminal_log), dtype=float), note

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
