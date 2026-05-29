"""Forward-return scenario buckets.

The six buckets partition fractional returns over a horizon. Convention: each
bucket is [lower, upper) — lower inclusive, upper exclusive — with ``None`` for
open ends. So a return of exactly -0.10 falls in "-10% to -5%", and +0.10 falls
in "> +10%".
"""

from __future__ import annotations

import numpy as np

from stock_agent.schemas.forecast import ProbBucket

# (label, lower, upper) as fractional returns.
DEFAULT_BUCKETS: list[tuple[str, float | None, float | None]] = [
    ("< -10%", None, -0.10),
    ("-10% to -5%", -0.10, -0.05),
    ("-5% to 0%", -0.05, 0.0),
    ("0% to +5%", 0.0, 0.05),
    ("+5% to +10%", 0.05, 0.10),
    ("> +10%", 0.10, None),
]


def assign_bucket(r: float) -> int:
    """Return the index of the bucket containing return ``r``."""
    for i, (_label, lo, hi) in enumerate(DEFAULT_BUCKETS):
        if (lo is None or r >= lo) and (hi is None or r < hi):
            return i
    return len(DEFAULT_BUCKETS) - 1  # defensive; ranges are exhaustive


def bucket_probabilities(returns: np.ndarray) -> list[float]:
    """Empirical probability mass per bucket from a sample of returns."""
    n = len(returns)
    if n == 0:
        return [0.0] * len(DEFAULT_BUCKETS)
    counts = [0] * len(DEFAULT_BUCKETS)
    for r in returns:
        counts[assign_bucket(float(r))] += 1
    return [c / n for c in counts]


def make_prob_buckets(probabilities: list[float]) -> list[ProbBucket]:
    """Pair probabilities with the bucket definitions into ``ProbBucket`` models."""
    return [
        ProbBucket(label=label, lower=lo, upper=hi, probability=p)
        for (label, lo, hi), p in zip(DEFAULT_BUCKETS, probabilities, strict=True)
    ]
