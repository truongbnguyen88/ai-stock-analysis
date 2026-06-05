"""Quantiles (VaR / CI) from a discrete scenario-bucket distribution.

A ``ScenarioForecast``'s buckets pin the CDF only at the finite bucket boundaries, and
the two outer buckets are **open** (no within-bucket shape). This reconstructs a
monotone CDF from the bucket masses — plus any known quantile anchors (e.g. a
sample-based model's own var/ci) — ramps the open tails linearly to 0 / 1 one
outer-band beyond the extremes, and inverts it.

So a quantile that lands *inside* an open outer bucket (common for deep VaR on volatile
names) is a **coarse approximation** — adequate for a VaR/CI readout, not a precise
extreme-tail estimate. Sample-based models (historical / Monte-Carlo) pass their exact
var/ci as anchors, so only models that expose buckets alone (ML) rely on the ramp.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stock_agent.schemas.forecast import ProbBucket

Anchor = tuple[float, float]  # (return, cumulative_probability)


def cdf_from_buckets(
    buckets: Sequence[ProbBucket], anchors: Sequence[Anchor] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a monotone CDF as sorted ``(returns, cumulative_prob)`` arrays.

    Anchors come from the finite bucket upper bounds (cumulative mass) plus any caller-
    supplied ``(return, cumprob)`` points; the open tails are ramped to 0 / 1 one
    outer-band beyond the extremes so the CDF spans [0, 1].
    """
    xs: list[float] = []
    fs: list[float] = []
    finite: list[float] = []
    cum = 0.0
    for b in buckets:
        cum += b.probability
        if b.upper is not None:
            xs.append(float(b.upper))
            fs.append(min(cum, 1.0))
            finite.append(abs(float(b.upper)))
        if b.lower is not None:
            finite.append(abs(float(b.lower)))
    if anchors:
        for ax, af in anchors:
            xs.append(float(ax))
            fs.append(float(af))
    band = max(finite) if finite else 0.1
    xs += [min(xs) - band, max(xs) + band]
    fs += [0.0, 1.0]
    order = np.argsort(np.asarray(xs))
    x = np.asarray(xs, dtype=float)[order]
    # np.asarray wraps the accumulate result so mypy types it as an array, not a scalar.
    f = np.asarray(np.maximum.accumulate(np.asarray(fs, dtype=float)[order]))  # monotone CDF
    keep = np.concatenate([np.diff(x) > 0, np.array([True])])  # de-dup x, keep cummax F
    return x[keep], f[keep]


def invert_cdf(xs: np.ndarray, fs: np.ndarray, levels: Sequence[float]) -> dict[float, float]:
    """Invert a monotone CDF at each probability level by linear interpolation."""
    return {lvl: float(np.interp(lvl, fs, xs)) for lvl in levels}


def quantiles_from_buckets(
    buckets: Sequence[ProbBucket],
    levels: Sequence[float],
    *,
    anchors: Sequence[Anchor] | None = None,
) -> dict[float, float]:
    """Quantiles of a bucket distribution at ``levels`` (see module docstring caveat)."""
    xs, fs = cdf_from_buckets(buckets, anchors)
    return invert_cdf(xs, fs, levels)
