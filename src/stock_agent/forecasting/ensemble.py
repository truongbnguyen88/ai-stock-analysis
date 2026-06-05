"""Ensemble forecaster — linear probability pool over validated members.

Rationale: the GARCH validation (Task 9) showed different *mechanisms* — empirical
simulation, block bootstrap, conditional-volatility GARCH — are complementary, and a
probability-pooled ensemble exploits that. Ensembles typically beat any single member
on Brier *without new information*, so this sidesteps the price-only information
ceiling that capped every single-model effort. See docs/validations_results.md.

Blending (mathematically principled — see the per-stat notes):
  - **bucket probabilities**: linear pool ``p_blend[i] = Σ_m w_m · p_m[i]`` — the exact
    mixture masses (all members share the horizon's bucket scheme).
  - **E[r], upside, downside**: exact weighted average — these are *linear* functionals
    of the distribution, so the mixture value IS the weighted mean of members' values.
  - **VaR / CI (quantiles)**: NON-linear, so we never average members' quantiles.
    Each member's CDF is reconstructed from its bucket-boundary cumulative masses + its
    own quantile anchors (var/ci); the mixture CDF ``Σ_m w_m · F_m`` is then inverted at
    the 1% / 5% / 95% levels.

v1 weights are **equal** (leakage-safe, no fitting). Skill-weighting (∝ 1/Brier from
the backtest harness, estimated on prior folds only) is a planned leakage-safe
follow-up — ``blend_forecasts`` already accepts arbitrary weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date

import numpy as np

from stock_agent.forecasting.base import ForecastModel
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.forecasting.monte_carlo import MonteCarlo
from stock_agent.forecasting.quantiles import cdf_from_buckets
from stock_agent.logging_config import get_logger
from stock_agent.schemas.forecast import ProbBucket, ScenarioForecast
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)

ENSEMBLE_NAME = "ensemble"
# Stateless (artifact-free) members. GBM is excluded (worst Brier; dropped from the
# agent) — these three are the promoted stateless forecasters.
STATELESS_MEMBER_NAMES: tuple[str, ...] = (
    "historical_sim",
    "monte_carlo_bootstrap",
    "monte_carlo_garch",
)
# An ML member that can't find its trained artifact falls back to historical_sim
# (keeping its own name) and tags the note with this marker. Such a member is a
# duplicate of the historical baseline, so the ensemble drops it rather than
# triple-counting historical.
_FALLBACK_MARKER = "Fell back to historical_sim"


def _normalize(weights: Sequence[float]) -> list[float]:
    """Non-negative weights renormalized to sum to 1 (equal if all zero)."""
    w = [max(0.0, float(x)) for x in weights]
    total = sum(w)
    if total <= 0:
        return [1.0 / len(w)] * len(w)
    return [x / total for x in w]


def _member_anchors(fc: ScenarioForecast) -> list[tuple[float, float]]:
    """The member's own quantile points (var/ci) as ``(return, cumprob)`` anchors.

    Sample-based members (historical / Monte-Carlo) and now ML all expose var/ci, so
    these sharpen the reconstructed CDF beyond the coarse bucket boundaries.
    """
    tail = (1.0 - fc.ci_level) / 2.0 if fc.ci_level is not None else 0.05
    pairs = ((fc.var_99, 0.01), (fc.var_95, 0.05), (fc.ci_low, tail), (fc.ci_high, 1.0 - tail))
    return [(float(v), lvl) for v, lvl in pairs if v is not None]


def _cdf_anchors(fc: ScenarioForecast) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a member's CDF (shared bucket→CDF logic; see forecasting.quantiles)."""
    return cdf_from_buckets(fc.buckets, _member_anchors(fc))


def _mixture_quantiles(
    anchors: list[tuple[np.ndarray, np.ndarray]], weights: Sequence[float], levels: Sequence[float]
) -> dict[float, float]:
    """Invert the weighted mixture CDF ``Σ w_m F_m`` at each level (never averages quantiles)."""
    grid = np.unique(np.concatenate([x for x, _ in anchors]))
    f_mix = np.zeros_like(grid)
    for (x, f), w in zip(anchors, weights, strict=True):
        # Outside a member's anchor range, clamp to its end masses (flat CDF tails).
        f_mix += w * np.interp(grid, x, f, left=float(f[0]), right=float(f[-1]))
    return {lvl: float(np.interp(lvl, f_mix, grid)) for lvl in levels}


def blend_forecasts(
    forecasts: Sequence[ScenarioForecast],
    weights: Sequence[float] | None = None,
    *,
    ticker: str,
    as_of: Date,
    horizon_days: int,
    member_names: Sequence[str] | None = None,
    model_name: str = ENSEMBLE_NAME,
) -> ScenarioForecast:
    """Linear-pool member forecasts into one ``ScenarioForecast`` (see module docstring)."""
    if not forecasts:
        raise ValueError("cannot blend an empty set of forecasts")
    n = len(forecasts)
    w = _normalize(weights if weights is not None else [1.0] * n)

    n_buckets = len(forecasts[0].buckets)
    if any(len(f.buckets) != n_buckets for f in forecasts):
        raise ValueError("members disagree on bucket count — cannot pool (horizon mismatch?)")

    template = forecasts[0].buckets
    blended = [
        ProbBucket(
            label=template[i].label,
            lower=template[i].lower,
            upper=template[i].upper,
            probability=sum(w[m] * forecasts[m].buckets[i].probability for m in range(n)),
        )
        for i in range(n_buckets)
    ]
    # Linear functionals → exact weighted means.
    exp_ret = sum(w[m] * forecasts[m].expected_return for m in range(n))
    upside = sum(w[m] * forecasts[m].upside_prob for m in range(n))
    downside = sum(w[m] * forecasts[m].downside_prob for m in range(n))
    # Quantiles via the mixture CDF.
    q = _mixture_quantiles([_cdf_anchors(f) for f in forecasts], w, levels=(0.01, 0.05, 0.95))

    names = list(member_names) if member_names is not None else [f.model_name for f in forecasts]
    weight_str = ", ".join(f"{nm} {wm:.2f}" for nm, wm in zip(names, w, strict=True))
    return ScenarioForecast(
        ticker=ticker,
        as_of=as_of,
        horizon_days=horizon_days,
        model_name=model_name,
        buckets=blended,
        expected_return=exp_ret,
        upside_prob=upside,
        downside_prob=downside,
        var_95=q[0.05],
        var_99=q[0.01],
        ci_level=0.90,
        ci_low=q[0.05],
        ci_high=q[0.95],
        calibration_status="unknown",
        notes=f"Equal-weight ensemble ({weight_str})",
    )


class EnsembleForecast:
    """``ForecastModel`` that linear-pools several member forecasters."""

    def __init__(
        self,
        members: Sequence[ForecastModel],
        *,
        weights: Sequence[float] | None = None,
        name: str = ENSEMBLE_NAME,
    ) -> None:
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = list(members)
        self.weights = list(weights) if weights is not None else None
        self.name = name

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        as_of = as_of or series.bars[-1].date
        forecasts: list[ScenarioForecast] = []
        names: list[str] = []
        used_w: list[float] = []
        for i, member in enumerate(self.members):
            try:
                fc = member.forecast(series, horizon_days=horizon_days, as_of=as_of)
            except Exception as exc:  # noqa: BLE001 — member isolation: one bad model never aborts
                log.warning("ensemble.member_failed", member=member.name, error=str(exc))
                continue
            if fc.notes and _FALLBACK_MARKER in fc.notes:
                # ML member with no artifact → degenerate duplicate of historical; drop it.
                log.warning("ensemble.member_fellback", member=member.name)
                continue
            forecasts.append(fc)
            names.append(member.name)
            used_w.append(self.weights[i] if self.weights is not None else 1.0)
        if not forecasts:
            raise ValueError("all ensemble members failed to produce a forecast")
        return blend_forecasts(
            forecasts,
            used_w,
            ticker=series.ticker,
            as_of=as_of,
            horizon_days=horizon_days,
            member_names=names,
            model_name=self.name,
        )


def _mc(variant: str, mc_paths: int | None) -> MonteCarlo:
    return MonteCarlo(variant=variant) if mc_paths is None else MonteCarlo(variant=variant, n_paths=mc_paths)  # type: ignore[arg-type]  # noqa: E501


def _stateless_members(mc_paths: int | None) -> list[ForecastModel]:
    return [HistoricalSimulation(), _mc("bootstrap", mc_paths), _mc("garch", mc_paths)]


def default_ensemble(*, mc_paths: int | None = None) -> EnsembleForecast:
    """The 3 stateless members (historical + bootstrap + GARCH), equal-weighted.

    Artifact-free and always available. ``mc_paths`` lets the backtest use fewer
    Monte-Carlo paths for speed.
    """
    return EnsembleForecast(_stateless_members(mc_paths))


def full_ensemble(registry: object, *, mc_paths: int | None = None) -> EnsembleForecast:
    """The 5-member ensemble = the 3 stateless members + pooled ML (logistic + lightgbm).

    ML members load their trained artifacts via the ``registry`` (also used for the
    earnings feature). When an ML artifact is absent, that member self-reports a
    historical fallback and is dropped at forecast time — so this gracefully reduces
    to :func:`default_ensemble` when models haven't been trained/pulled.
    """
    from stock_agent.forecasting.ml import MLForecaster

    members = _stateless_members(mc_paths)
    members += [
        MLForecaster("logistic", registry=registry),  # type: ignore[arg-type]
        MLForecaster("lightgbm", registry=registry),  # type: ignore[arg-type]
    ]
    return EnsembleForecast(members)
