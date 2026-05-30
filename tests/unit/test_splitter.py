"""Walk-forward splitter: fold structure, embargo, and leakage invariants."""

from __future__ import annotations

import pytest

from stock_agent.backtesting.splitter import Fold, assert_no_leakage, walk_forward_splits


def test_expanding_folds_basic_structure() -> None:
    folds = walk_forward_splits(
        n_bars=100, horizon=5, min_train=20, test_size=3, stride=5, embargo=5
    )
    f0 = folds[0]
    # First test as-of = min_train + embargo = 25; train_end = 25 - 5 - 1 = 19.
    assert f0.train_start == 0  # expanding
    assert f0.train_end == 19
    assert f0.test_as_of == (25, 30, 35)
    # Folds advance one stride past the previous test block.
    assert folds[1].test_start == 40


def test_embargo_prevents_target_overlap() -> None:
    folds = walk_forward_splits(n_bars=200, horizon=10, min_train=30, test_size=4)
    # Default stride = embargo = horizon. The last training target (train_end+h)
    # must end strictly before the first test as-of.
    for f in folds:
        assert f.train_end + 10 < f.test_start
    assert_no_leakage(folds, horizon=10)  # invariant holds


def test_test_windows_disjoint_and_increasing() -> None:
    folds = walk_forward_splits(n_bars=300, horizon=20, min_train=60, test_size=3)
    seen: list[int] = []
    for f in folds:
        seen.extend(f.test_as_of)
    assert seen == sorted(seen)
    assert len(seen) == len(set(seen))  # no repeats across folds


def test_rolling_window_limits_train_start() -> None:
    folds = walk_forward_splits(
        n_bars=100,
        horizon=5,
        min_train=20,
        test_size=3,
        stride=5,
        embargo=5,
        expanding=False,
        rolling_window=10,
    )
    # train_end=19 → rolling train_start = 19 - 10 + 1 = 10 (not 0).
    assert folds[0].train_start == 10
    assert folds[0].train_end == 19


def test_raises_when_series_too_short() -> None:
    with pytest.raises(ValueError, match="too short"):
        walk_forward_splits(n_bars=30, horizon=20, min_train=20, test_size=3)


def test_rolling_requires_window() -> None:
    with pytest.raises(ValueError, match="rolling_window"):
        walk_forward_splits(n_bars=100, horizon=5, min_train=20, test_size=3, expanding=False)


def test_assert_no_leakage_catches_violation() -> None:
    # Hand-build a leaky fold (test starts right after train, no embargo).
    bad = [Fold(index=0, train_start=0, train_end=50, test_as_of=(51, 52))]
    with pytest.raises(AssertionError, match="embargo"):
        assert_no_leakage(bad, horizon=20)
