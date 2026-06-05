"""Split-conformal calibration of prediction intervals (distribution-free coverage).

A forecaster's CI (``ci_low``/``ci_high`` at, say, 90%) is the *model's* believed
interval — out-of-sample it may cover more or less than 90%. Split conformal turns it
into a **guarantee**: from a held-out set of (predicted-interval, realized) pairs it
computes a single correction ``q`` such that the adjusted interval ``[lo - q, hi + q]``
has **>= 1 - alpha marginal coverage** on exchangeable future data — no distributional
assumptions, finite-sample valid (Vovk; CQR, Romano et al. 2019).

Mechanics:
  - **Nonconformity score** (CQR): ``E = max(lo - y, y - hi)`` — negative when the
    realized ``y`` is inside the interval (distance to the nearest edge), positive when
    outside (how far). So large E = the interval missed.
  - **Correction** ``q`` = the finite-sample ``(1 - alpha)`` quantile of the calibration
    scores: the ``ceil((n+1)(1-alpha))``-th smallest. ``q > 0`` widens an under-covering
    interval; ``q < 0`` tightens an over-covering one.
  - **Guarantee** is *marginal* (averaged over the calibration distribution), not
    conditional per-ticker — the standard split-conformal caveat.

Leakage: the calibration pairs must come strictly from data available before the
forecast as-of (and embargoed by the horizon to avoid overlapping targets).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Accept python sequences or numpy arrays at the coverage helpers.
ArrayLike = Sequence[float] | np.ndarray


def nonconformity(lower: float, upper: float, y: float) -> float:
    """CQR score: how far ``y`` fell outside ``[lower, upper]`` (negative if inside)."""
    return max(lower - y, y - upper)


def conformal_correction(scores: Sequence[float], *, alpha: float) -> float:
    """Finite-sample ``(1 - alpha)`` quantile of calibration scores (the interval shift ``q``).

    Returns the ``ceil((n+1)(1-alpha))``-th smallest score. When that rank exceeds ``n``
    (too few points for the requested coverage), returns ``+inf`` — i.e. no finite
    interval can honor the guarantee, so the caller should widen to unbounded or collect
    more calibration data. ``alpha`` in (0, 1); e.g. ``alpha=0.1`` for 90% coverage.
    """
    n = len(scores)
    if n == 0:
        return math.inf
    rank = math.ceil((n + 1) * (1.0 - alpha))  # 1-indexed
    if rank > n:
        return math.inf
    return float(np.sort(np.asarray(scores, dtype=float))[rank - 1])


def conformalize_interval(lower: float, upper: float, q: float) -> tuple[float, float]:
    """Apply the correction: the conformalized interval ``[lower - q, upper + q]``.

    A tightening (``q < 0``) is capped so the interval never inverts (``lo <= hi``).
    """
    lo, hi = lower - q, upper + q
    if lo > hi:  # over-tightened (q very negative) → collapse to the midpoint
        mid = 0.5 * (lower + upper)
        return mid, mid
    return lo, hi


def empirical_coverage(lowers: ArrayLike, uppers: ArrayLike, ys: ArrayLike) -> float:
    """Fraction of realized ``ys`` that fell within their ``[lower, upper]`` interval."""
    lo = np.asarray(lowers, dtype=float)
    hi = np.asarray(uppers, dtype=float)
    y = np.asarray(ys, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((y >= lo) & (y <= hi)))


@dataclass(frozen=True)
class ConformalMetrics:
    """Coverage diagnostics for a model's CI over an OOS run (honest temporal split)."""

    alpha: float  # 1 - nominal coverage (e.g. 0.10 for a 90% CI)
    n_eval: int  # points the coverages are measured on (the later/test half)
    empirical_coverage: float  # what the model's STATED CI actually covered (test half)
    correction: float  # q, fit on the earlier/calibration half (inf if too few points)
    conformalized_coverage: float | None  # coverage of [lo-q, hi+q] on the test half
    mean_width: float  # mean stated interval width (test half)
    mean_width_conformal: float | None  # mean conformalized width (test half)


def conformal_metrics(
    lowers: Sequence[float],
    uppers: Sequence[float],
    ys: Sequence[float],
    *,
    alpha: float,
    holdout_frac: float = 0.5,
    min_each: int = 15,
) -> ConformalMetrics | None:
    """Split-conformal coverage diagnostics, chronological (no shuffling = no leakage).

    Fit the correction ``q`` on the earlier ``1 - holdout_frac`` of the run, then compare
    the model's *stated* CI coverage to the *conformalized* coverage on the later held-out
    part. With too few points to split, reports stated coverage on all points and leaves
    the conformalized fields ``None``. Returns ``None`` if there are no points at all.
    """
    lo = np.asarray(lowers, dtype=float)
    hi = np.asarray(uppers, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = y.size
    if n == 0:
        return None

    cut = int(round(n * (1.0 - holdout_frac)))
    if cut < min_each or (n - cut) < min_each:  # not enough to split honestly
        return ConformalMetrics(
            alpha=alpha, n_eval=n, empirical_coverage=empirical_coverage(lo, hi, y),
            correction=math.nan, conformalized_coverage=None,
            mean_width=float(np.mean(hi - lo)), mean_width_conformal=None,
        )

    scores = [nonconformity(float(lo[i]), float(hi[i]), float(y[i])) for i in range(cut)]
    q = conformal_correction(scores, alpha=alpha)
    lo_t, hi_t, y_t = lo[cut:], hi[cut:], y[cut:]
    emp = empirical_coverage(lo_t, hi_t, y_t)
    if math.isfinite(q):
        a, b = np.minimum(lo_t - q, hi_t + q), np.maximum(lo_t - q, hi_t + q)
        conf_cov: float | None = float(np.mean((y_t >= a) & (y_t <= b)))
        width_c: float | None = float(np.mean(b - a))
    else:
        conf_cov, width_c = None, None
    return ConformalMetrics(
        alpha=alpha, n_eval=int(n - cut), empirical_coverage=emp, correction=q,
        conformalized_coverage=conf_cov, mean_width=float(np.mean(hi_t - lo_t)),
        mean_width_conformal=width_c,
    )
