"""Offline pooled conformal calibration of prediction intervals (calibrate / serve).

Goal: a CI you can trust. Each served forecaster's CI is corrected by a single
distribution-free ``q`` per ``(model, horizon)`` so the served ``[lo - q, hi + q]``
has ~the nominal coverage out-of-sample.

How ``q`` is estimated (leakage-safe, bounded — see module decision in docs):
  - Build each model **as of a calibration cutoff** ``T_cal`` (ML trained on data
    ``<= T_cal``; stateless models are themselves). This trains ML ONCE per
    (model, horizon), not per fold.
  - Forecast every basket ticker point-in-time across the held-out ``(T_cal, end-h]``
    window → pool the ``(ci_low, ci_high, realized)`` triples across tickers →
    ``q = conformal_correction(scores, alpha = 1 - ci_level)``.
  - The pool is large (basket × window), so ``q`` is stable where a single-ticker fit
    was noisy. The holdout is strictly in the past, so this is leakage-safe; the served
    model is trained on more recent data, so ``q`` is a (documented, re-validated)
    correction for the model *class*'s structural CI miscoverage, not an exact match.

Persisted as ``outputs/models/conformal.json`` and applied at inference by
``pipelines.forecast`` (config-gated ``settings.conformal_intervals``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime
from pathlib import Path

from stock_agent.forecasting.conformal import (
    conformal_correction,
    empirical_coverage,
    nonconformity,
)
from stock_agent.logging_config import get_logger
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)

CONFORMAL_FILE = "conformal.json"


@dataclass(frozen=True)
class ConformalEntry:
    q: float
    n: int
    coverage_before: float
    coverage_after: float


@dataclass
class ConformalArtifact:
    """Per-(model, horizon) conformal corrections at one nominal CI level."""

    ci_level: float
    cal_cutoff: str
    calibrated_at: str
    entries: dict[str, dict[int, ConformalEntry]] = field(default_factory=dict)

    def q_for(self, model_name: str, horizon_days: int) -> float | None:
        """The correction for a model+horizon, or None if not calibrated."""
        e = self.entries.get(model_name, {}).get(horizon_days)
        return e.q if e is not None else None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "ci_level": self.ci_level,
            "cal_cutoff": self.cal_cutoff,
            "calibrated_at": self.calibrated_at,
            "entries": {
                m: {str(h): vars(e) for h, e in by_h.items()} for m, by_h in self.entries.items()
            },
        }
        path.write_text(json.dumps(blob, indent=2))

    @staticmethod
    def load(path: Path) -> ConformalArtifact | None:
        if not path.exists():
            return None
        blob = json.loads(path.read_text())
        entries = {
            m: {int(h): ConformalEntry(**e) for h, e in by_h.items()}
            for m, by_h in blob.get("entries", {}).items()
        }
        return ConformalArtifact(
            ci_level=blob["ci_level"],
            cal_cutoff=blob["cal_cutoff"],
            calibrated_at=blob["calibrated_at"],
            entries=entries,
        )


def interval_scores(
    model: object,
    series: PriceSeries,
    horizon_days: int,
    *,
    cal_cutoff: Date,
    stride: int = 1,
) -> list[tuple[float, float, float]]:
    """Point-in-time ``(ci_low, ci_high, realized)`` for as-ofs strictly after ``cal_cutoff``.

    Forecasts from ``bars[:t+1]`` at each ``t`` whose date > cal_cutoff and that has a
    realized ``t + horizon`` bar — so the model only ever sees data up to ``t``. ``stride``
    subsamples as-ofs (e.g. forecast every ``stride`` bars) to bound the cost.
    """
    closes = series.closes
    dates = series.dates
    n = len(series)
    out: list[tuple[float, float, float]] = []
    for t in range(0, n - horizon_days, max(1, stride)):
        if dates[t] <= cal_cutoff:
            continue
        sub = PriceSeries(ticker=series.ticker, bars=series.bars[: t + 1])
        try:
            fc: ScenarioForecast = model.forecast(  # type: ignore[attr-defined]
                sub, horizon_days=horizon_days, as_of=dates[t]
            )
        except (ValueError, RuntimeError):
            continue
        if fc.ci_low is None or fc.ci_high is None:
            continue
        realized = float(closes[t + horizon_days] / closes[t] - 1.0)
        out.append((float(fc.ci_low), float(fc.ci_high), realized))
    return out


def fit_entry(
    triples: list[tuple[float, float, float]], *, ci_level: float
) -> ConformalEntry | None:
    """Pool the (CI, realized) triples → one conformal correction + before/after coverage."""
    if not triples:
        return None
    lows = [a for a, _, _ in triples]
    highs = [b for _, b, _ in triples]
    ys = [y for _, _, y in triples]
    scores = [nonconformity(a, b, y) for a, b, y in triples]
    q = conformal_correction(scores, alpha=1.0 - ci_level)
    cov_before = empirical_coverage(lows, highs, ys)
    n = len(ys)
    if q == float("inf"):  # too few points for the guarantee → no correction
        return ConformalEntry(q=0.0, n=n, coverage_before=cov_before, coverage_after=cov_before)
    cov_after = empirical_coverage([a - q for a in lows], [b + q for b in highs], ys)
    return ConformalEntry(q=float(q), n=n, coverage_before=cov_before, coverage_after=cov_after)


def new_artifact(ci_level: float, cal_cutoff: Date) -> ConformalArtifact:
    return ConformalArtifact(
        ci_level=ci_level,
        cal_cutoff=cal_cutoff.isoformat(),
        calibrated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )
