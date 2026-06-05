"""Tests for bucket→quantile reconstruction (VaR/CI from a discrete distribution)."""

from __future__ import annotations

from stock_agent.forecasting.buckets import buckets_for_horizon, make_prob_buckets
from stock_agent.forecasting.quantiles import cdf_from_buckets, quantiles_from_buckets
from stock_agent.schemas.forecast import ProbBucket


def _buckets(probs: list[float]) -> list[ProbBucket]:
    return make_prob_buckets(probs, buckets_for_horizon(20))  # boundaries ±5/±10%


def test_quantiles_are_ordered_and_in_range() -> None:
    # Roughly symmetric, fat-ish distribution.
    q = quantiles_from_buckets(_buckets([0.05, 0.10, 0.35, 0.35, 0.10, 0.05]), (0.01, 0.05, 0.95))
    assert q[0.01] < q[0.05] < q[0.95]  # monotone
    assert q[0.01] < 0 < q[0.95]  # tails straddle zero


def test_deep_tail_lands_in_open_bucket_below_outer_boundary() -> None:
    # 7% mass below -10% → the 5% VaR sits in the open "< -10%" bucket (below -0.10).
    q = quantiles_from_buckets(_buckets([0.07, 0.10, 0.33, 0.33, 0.12, 0.05]), (0.05,))
    assert q[0.05] < -0.10


def test_cdf_spans_zero_to_one_and_is_monotone() -> None:
    xs, fs = cdf_from_buckets(_buckets([0.05, 0.15, 0.30, 0.30, 0.15, 0.05]))
    assert fs[0] == 0.0 and fs[-1] == 1.0
    assert all(b >= a for a, b in zip(fs, fs[1:], strict=False))  # non-decreasing
    assert all(b > a for a, b in zip(xs, xs[1:], strict=False))  # strictly increasing returns


def test_anchors_sharpen_the_tail() -> None:
    # Without an anchor the deep tail relies on the ramp; an explicit var_99 anchor moves it.
    b = _buckets([0.05, 0.10, 0.35, 0.35, 0.10, 0.05])
    plain = quantiles_from_buckets(b, (0.01,))[0.01]
    anchored = quantiles_from_buckets(b, (0.01,), anchors=[(-0.25, 0.01)])[0.01]
    assert abs(anchored - (-0.25)) < 1e-9
    assert plain != anchored
