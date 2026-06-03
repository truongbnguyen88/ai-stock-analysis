"""Regime-conditional historical-simulation forecaster (Task 8 spike).

Fits a Gaussian HMM on the stock's own daily ``(log-return, |log-return|)``
sequence using ONLY data up to ``as_of``, identifies the current latent regime,
and builds the forward-return distribution from the *historical h-day returns
whose start-day was in that same regime*. The motivation is the recurring
validation finding: direction is ≈ efficient, but **volatility/magnitude is
predictable** — and latent regimes are volatility states, so conditioning on the
current regime sharpens exactly the big-move tail the toolkit already targets.

Why it is leakage-safe: the HMM is fit on observations derived from
``closes[: as_of]`` only, and every conditioned forward return is realized
strictly within ``[0, as_of]`` (start day ``i`` with ``i + h <= t``). No price
data at or after ``as_of`` is touched. The Viterbi state labels use
within-window smoothing, which is fine — we use them to *characterize* a
historical conditional distribution, not to predict.

Status — EXPERIMENTAL. Reachable via the ``forecast`` / ``backtest`` CLI for
evaluation against the shipped toolkit; deliberately NOT wired into the agent or
the written report until it earns promotion on the same disjoint-basket
Brier/ECE discipline as every other model (see docs/TASKS.md, Task 8).
"""

from __future__ import annotations

from datetime import date as Date

import numpy as np

from stock_agent.forecasting.historical import _horizon_returns, sample_to_forecast
from stock_agent.logging_config import get_logger
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)

_MODEL_NAME = "regime_hmm"
_MIN_FIT_BARS = 252  # ~1y of daily observations before regimes are meaningful
_MIN_REGIME_SAMPLES = 30  # conditioned forward returns needed, else fall back


def _daily_observations(closes: np.ndarray) -> np.ndarray:
    """2-D daily HMM observation sequence: ``(log-return, |log-return|)``.

    ``|log-return|`` is a scale-free volatility proxy so the latent states
    separate calm vs. stressed regimes (the predictable axis). Row ``k`` is the
    day-``k`` → day-``k+1`` transition, i.e. it describes day ``k + 1``.
    """
    logret = np.diff(np.log(closes))
    return np.column_stack([logret, np.abs(logret)])


class RegimeForecaster:
    """``ForecastModel``: regime-conditional empirical forward-return distribution."""

    def __init__(
        self,
        *,
        n_states: int = 3,
        random_state: int = 42,
        min_fit_bars: int = _MIN_FIT_BARS,
        min_regime_samples: int = _MIN_REGIME_SAMPLES,
    ) -> None:
        self.name = _MODEL_NAME
        self._n_states = n_states
        self._random_state = random_state  # fixes HMM init → deterministic
        self._min_fit_bars = min_fit_bars
        self._min_regime_samples = min_regime_samples

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        as_of = as_of or series.bars[-1].date
        closes = np.asarray(series.closes, dtype=float)

        # Need enough history to (a) fit regimes and (b) have conditioned forward
        # returns; otherwise fall back to the unconditional baseline.
        if len(closes) < self._min_fit_bars + horizon_days:
            return self._fallback(series, horizon_days, as_of, "insufficient history for regimes")

        states = self._fit_states(closes, ticker=series.ticker)
        if states is None:
            return self._fallback(series, horizon_days, as_of, "HMM fit unavailable")

        # Forward h-day returns; fwd[i] starts at day i (completes at i+h <= t).
        fwd = _horizon_returns(closes, horizon_days)
        if len(fwd) <= 1:
            return self._fallback(series, horizon_days, as_of, "no forward returns")

        # Regime at start-day i is states[i-1] (states[k] describes day k+1). Select
        # the start days that were in the *current* regime (states[-1]).
        current = int(states[-1])
        starts = np.arange(1, len(fwd))
        in_regime = states[starts - 1] == current
        sample = fwd[starts[in_regime]]

        if len(sample) < self._min_regime_samples:
            return self._fallback(
                series,
                horizon_days,
                as_of,
                f"regime {current} had only {len(sample)} conditioned samples",
            )

        fc = sample_to_forecast(
            sample,
            ticker=series.ticker,
            as_of=as_of,
            horizon_days=horizon_days,
            model_name=self.name,
        )
        regime_note = (
            f"Conditioned on regime {current}/{self._n_states} "
            f"({len(sample)} of {len(fwd)} historical windows)."
        )
        note = f"{fc.notes} {regime_note}" if fc.notes else regime_note
        return fc.model_copy(update={"notes": note})

    def _fit_states(self, closes: np.ndarray, *, ticker: str) -> np.ndarray | None:
        """Fit the Gaussian HMM on data <= as_of and return the Viterbi state path.

        Returns ``None`` on any failure (missing dep, non-convergence, singular
        covariance) so the caller falls back to the unconditional baseline.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            log.info("regime.hmmlearn_missing", ticker=ticker)
            return None
        obs = _daily_observations(closes)
        try:
            hmm = GaussianHMM(
                n_components=self._n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=self._random_state,
            )
            hmm.fit(obs)
            return np.asarray(hmm.predict(obs), dtype=int)
        except Exception as exc:  # noqa: BLE001 - any fit failure → safe fallback
            log.info("regime.fit_failed", ticker=ticker, error=str(exc))
            return None

    def _fallback(
        self, series: PriceSeries, horizon_days: int, as_of: Date, reason: str
    ) -> ScenarioForecast:
        """Unconditional historical-sim forecast, tagged as a regime fallback."""
        closes = np.asarray(series.closes, dtype=float)
        sample = _horizon_returns(closes, horizon_days)
        if len(sample) == 0:
            raise ValueError(
                f"insufficient price history ({len(closes)} bars) for horizon {horizon_days}"
            )
        fc = sample_to_forecast(
            sample,
            ticker=series.ticker,
            as_of=as_of,
            horizon_days=horizon_days,
            model_name=self.name,
        )
        note = f"Regime fallback ({reason}); used unconditional history."
        note = f"{fc.notes} {note}" if fc.notes else note
        return fc.model_copy(update={"notes": note})
