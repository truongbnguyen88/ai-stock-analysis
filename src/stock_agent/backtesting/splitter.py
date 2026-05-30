"""Walk-forward temporal splitting with an embargo (leakage prevention).

This is the correctness backbone of the backtest: it decides, for each fold,
which bar is the training cutoff and which bars are evaluated out-of-sample.

Conventions (all in *bar index* space, 0-based):
- An **as-of** point ``t`` is evaluable only if a realized future exists, i.e.
  ``t + horizon <= n_bars - 1`` (we can observe ``close[t+horizon]``).
- **Embargo** ``= horizon``: a gap between the training cutoff ``train_end`` and
  the first test as-of, so the last training row's target window ``[train_end,
  train_end+h]`` ends strictly before the first test as-of. This is the standard
  purge that prevents target-overlap leakage at the train/test boundary.
- **Stride** ``= horizon`` by default: spacing between consecutive test as-of
  points, so their h-day target windows do not overlap (reduces the
  autocorrelation that overlapping windows induce).
- **Expanding** training by default (train grows each fold); ``rolling_window``
  switches to a fixed-length trailing window (used by per-fold ML refits).

The split is purely structural (no data, no models) → deterministic and unit
testable. ``assert_no_leakage`` encodes the invariant the tests assert.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, in bar-index space.

    ``train_start..train_end`` (inclusive) is the data a refittable model may
    learn from; ``test_as_of`` are the evaluated as-of bar indices, each with a
    realized ``+horizon`` future. Stateless forecasters ignore the train range
    and simply forecast from ``bars[:t+1]`` at each ``t`` in ``test_as_of``.
    """

    index: int
    train_start: int
    train_end: int  # inclusive training cutoff (last bar a model may see)
    test_as_of: tuple[int, ...]

    @property
    def test_start(self) -> int:
        return self.test_as_of[0]

    @property
    def test_end(self) -> int:
        return self.test_as_of[-1]


def walk_forward_splits(
    *,
    n_bars: int,
    horizon: int,
    min_train: int,
    test_size: int,
    stride: int | None = None,
    embargo: int | None = None,
    expanding: bool = True,
    rolling_window: int | None = None,
) -> list[Fold]:
    """Build expanding/rolling walk-forward folds with an embargo.

    Args:
        n_bars: total number of price bars.
        horizon: forecast horizon ``h`` in bars (sets default stride + embargo).
        min_train: minimum training bars before the first test as-of.
        test_size: number of test as-of points per fold.
        stride: spacing (bars) between test as-of points (default ``horizon``).
        embargo: gap (bars) between ``train_end`` and the first test as-of
            (default ``horizon``).
        expanding: expanding training window if True, else rolling.
        rolling_window: trailing window length (bars) when ``expanding`` is False.

    Returns:
        Folds in chronological order; their ``test_as_of`` sets are disjoint and
        strictly increasing.

    Raises:
        ValueError: on invalid params or when the series is too short for a fold.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1 (got {horizon})")
    if min_train < 1 or test_size < 1:
        raise ValueError("min_train and test_size must be >= 1")
    stride = horizon if stride is None else stride
    embargo = horizon if embargo is None else embargo
    if stride < 1 or embargo < 0:
        raise ValueError("stride must be >= 1 and embargo >= 0")
    if not expanding and (rolling_window is None or rolling_window < 1):
        raise ValueError("rolling_window (>=1) is required when expanding=False")

    # Last bar index that can serve as an as-of (has a realized +horizon future).
    last_as_of = n_bars - horizon - 1
    if last_as_of < min_train + embargo:
        raise ValueError(
            f"series too short: need > {min_train + embargo + horizon} bars, have {n_bars}"
        )

    folds: list[Fold] = []
    test_start = min_train + embargo  # first evaluable test as-of
    idx = 0
    while test_start <= last_as_of:
        # Up to test_size as-of points, spaced by stride, not past the last as-of.
        test_as_of = tuple(t for t in range(test_start, last_as_of + 1, stride))[:test_size]
        if not test_as_of:
            break
        train_end = test_start - embargo - 1  # inclusive; embargo bars before test
        train_start = 0 if expanding else max(0, train_end - rolling_window + 1)  # type: ignore[operator]
        folds.append(Fold(idx, train_start, train_end, test_as_of))
        idx += 1
        # Next fold starts one stride past this fold's last test point.
        test_start = test_as_of[-1] + stride

    if not folds:
        raise ValueError("no walk-forward folds could be built from the given parameters")
    return folds


def assert_no_leakage(folds: list[Fold], *, horizon: int) -> None:
    """Assert the temporal invariants (used by tests; cheap enough to keep on).

    For every fold: the last training target window ends strictly before the
    first test as-of (embargo respected), train precedes test, and test windows
    across folds neither overlap nor go backwards.
    """
    prev_test_end = -1
    for f in folds:
        assert f.train_end < f.test_start, f"fold {f.index}: train_end >= test_start"
        # Embargo: train_end's target peeks to train_end+horizon; must predate test.
        assert f.train_end + horizon < f.test_start, (
            f"fold {f.index}: embargo violated "
            f"(train_end {f.train_end} + h {horizon} >= test_start {f.test_start})"
        )
        assert f.test_start > prev_test_end, f"fold {f.index}: test windows overlap/regress"
        assert list(f.test_as_of) == sorted(f.test_as_of), f"fold {f.index}: test not sorted"
        prev_test_end = f.test_end
