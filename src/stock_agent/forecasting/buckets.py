"""Forward-return scenario buckets (horizon-scaled).

Six buckets partition fractional returns over a horizon. Convention: each bucket
is [lower, upper) — lower inclusive, upper exclusive — with ``None`` for open ends.

The boundaries scale with horizon, because a fixed ±5%/±10% band is informative at
20 days but degenerate at 60 (a 5% move over 60 days is near-certain — base rate
~90%). Each horizon uses a symmetric (inner, outer) band; the inner boundary is the
natural "big upside" threshold for that horizon (5% / 10% / 15%), the outer is 2×:

    h20 -> ±5%, ±10%   |   h30 -> ±10%, ±20%   |   h60 -> ±15%, ±30%

See docs/validations_results.md for the base-rate evidence that motivated this.
"""

from __future__ import annotations

import numpy as np

from stock_agent.schemas.forecast import ProbBucket

BucketDef = tuple[str, float | None, float | None]  # (label, lower, upper)

# (inner, outer) symmetric boundaries as fractional returns, per target horizon.
_HORIZON_BANDS: dict[int, tuple[float, float]] = {
    20: (0.05, 0.10),
    30: (0.10, 0.20),
    60: (0.15, 0.30),
}


def _bands_for(horizon_days: int) -> tuple[float, float]:
    """The (inner, outer) band for a horizon; nearest configured horizon if unlisted."""
    if horizon_days in _HORIZON_BANDS:
        return _HORIZON_BANDS[horizon_days]
    nearest = min(_HORIZON_BANDS, key=lambda h: abs(h - horizon_days))
    return _HORIZON_BANDS[nearest]


def buckets_for_horizon(horizon_days: int) -> list[BucketDef]:
    """The six (label, lower, upper) buckets for a horizon, scaled to its band."""
    inner, outer = _bands_for(horizon_days)
    i_pct, o_pct = round(inner * 100), round(outer * 100)
    return [
        (f"< -{o_pct}%", None, -outer),
        (f"-{o_pct}% to -{i_pct}%", -outer, -inner),
        (f"-{i_pct}% to 0%", -inner, 0.0),
        (f"0% to +{i_pct}%", 0.0, inner),
        (f"+{i_pct}% to +{o_pct}%", inner, outer),
        (f"> +{o_pct}%", outer, None),
    ]


def bucket_boundaries(buckets: list[BucketDef]) -> list[float]:
    """Sorted interior cut-points of a bucket scheme (e.g. -0.10, -0.05, 0, +0.05, +0.10).

    These are the return thresholds the pooled classifier trains on, and the set of
    valid ``k`` for the large-move reading.
    """
    edges = {b[1] for b in buckets if b[1] is not None}
    edges |= {b[2] for b in buckets if b[2] is not None}
    return sorted(edges)


def thresholds_for_horizon(horizon_days: int) -> list[float]:
    """Interior return cut-points the pooled classifier trains on for a horizon.

    e.g. h20 -> [-0.10, -0.05, 0, +0.05, +0.10]; h60 -> [-0.30, -0.15, 0, +0.15, +0.30].
    """
    return bucket_boundaries(buckets_for_horizon(horizon_days))


# Backward-compatible default = the 20-day scheme (the original ±5%/±10% buckets).
DEFAULT_BUCKETS: list[BucketDef] = buckets_for_horizon(20)


def assign_bucket(r: float, buckets: list[BucketDef] = DEFAULT_BUCKETS) -> int:
    """Return the index of the bucket containing return ``r``."""
    for i, (_label, lo, hi) in enumerate(buckets):
        if (lo is None or r >= lo) and (hi is None or r < hi):
            return i
    return len(buckets) - 1  # defensive; ranges are exhaustive


def bucket_probabilities(
    returns: np.ndarray, buckets: list[BucketDef] = DEFAULT_BUCKETS
) -> list[float]:
    """Empirical probability mass per bucket from a sample of returns."""
    n = len(returns)
    if n == 0:
        return [0.0] * len(buckets)
    counts = [0] * len(buckets)
    for r in returns:
        counts[assign_bucket(float(r), buckets)] += 1
    return [c / n for c in counts]


def make_prob_buckets(
    probabilities: list[float], buckets: list[BucketDef] = DEFAULT_BUCKETS
) -> list[ProbBucket]:
    """Pair probabilities with the bucket definitions into ``ProbBucket`` models."""
    return [
        ProbBucket(label=label, lower=lo, upper=hi, probability=p)
        for (label, lo, hi), p in zip(buckets, probabilities, strict=True)
    ]
