"""Brief-consolidation signal helpers in the research pipeline.

Locks the pure derivations the executive brief adds so ``research_summary`` carries the
same numbers the separate tools used to: the recent price snapshot (``_price_snapshot``)
and the shortest-horizon large-move tail split (``_primary_large_move``).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from stock_agent.pipelines.research import _price_snapshot, _primary_large_move
from stock_agent.schemas.forecast import ProbBucket, ScenarioForecast
from stock_agent.schemas.market import PriceBar, PriceSeries

_AS_OF = date(2026, 7, 2)


def _bar(d: date, close: float) -> PriceBar:
    # Flat OHLC around the close keeps the OHLC sanity validator happy.
    return PriceBar(date=d, open=close, high=close, low=close, close=close)


def _series(closes: list[float]) -> PriceSeries:
    # One trading day apart, ending at _AS_OF (so the last N days fall inside a 30d window).
    n = len(closes)
    bars = [_bar(_AS_OF - timedelta(days=n - 1 - i), c) for i, c in enumerate(closes)]
    return PriceSeries(ticker="MU", bars=bars)


def _forecast(horizon: int, boundary: float) -> ScenarioForecast:
    """A 4-bucket forecast with an outer +/-boundary so large_move has a valid k."""
    return ScenarioForecast(
        ticker="MU", as_of=_AS_OF, horizon_days=horizon, model_name="ensemble",
        buckets=[
            ProbBucket(label="big down", lower=None, upper=-boundary, probability=0.25),
            ProbBucket(label="down", lower=-boundary, upper=0.0, probability=0.25),
            ProbBucket(label="up", lower=0.0, upper=boundary, probability=0.20),
            ProbBucket(label="big up", lower=boundary, upper=None, probability=0.30),
        ],
        expected_return=0.02, upside_prob=0.5, downside_prob=0.5, var_95=-0.1,
    )


# ---- price snapshot ----------------------------------------------------------
def test_price_snapshot_window_and_returns() -> None:
    # 5 recent bars, all within 30 days of _AS_OF.
    snap = _price_snapshot(_series([100.0, 110.0, 90.0, 105.0, 102.0]))
    assert snap.n_bars == 5 and snap.window_days == 30
    assert snap.first_close == 100.0 and snap.last_close == 102.0
    assert snap.period_high == 110.0 and snap.period_low == 90.0
    assert snap.pct_change == pytest.approx(102.0 / 100.0 - 1.0)  # last/first over window
    assert snap.last_return == pytest.approx(102.0 / 105.0 - 1.0)  # last single-session move


def test_price_snapshot_excludes_bars_outside_window() -> None:
    # An old bar far outside 30 days must not enter the snapshot (window is trailing).
    old = _bar(_AS_OF - timedelta(days=200), 10.0)
    recent = [_bar(_AS_OF - timedelta(days=d), c) for d, c in ((3, 100.0), (2, 101.0), (0, 103.0))]
    series = PriceSeries(ticker="MU", bars=[old, *recent])
    snap = _price_snapshot(series)
    assert snap.n_bars == 3  # the 200-day-old bar is excluded
    assert snap.period_low == 100.0  # not the old 10.0


# ---- primary large move ------------------------------------------------------
def test_primary_large_move_uses_shortest_horizon_and_smallest_boundary() -> None:
    # Two horizons; the 20d (shortest) with a 0.05 boundary is the primary signal.
    lm = _primary_large_move([_forecast(60, 0.15), _forecast(20, 0.05)])
    assert lm is not None
    assert lm.horizon_days == 20 and lm.threshold == pytest.approx(0.05)
    assert lm.prob_big_up == pytest.approx(0.30) and lm.prob_big_down == pytest.approx(0.25)
    assert lm.prob_large_move == pytest.approx(0.55)


def test_primary_large_move_none_when_no_forecasts() -> None:
    assert _primary_large_move([]) is None


def test_primary_large_move_none_without_positive_boundary() -> None:
    # A degenerate up/down-only forecast has no positive bucket edge → no valid k.
    fc = ScenarioForecast(
        ticker="MU", as_of=_AS_OF, horizon_days=20, model_name="ensemble",
        buckets=[
            ProbBucket(label="down", lower=None, upper=0.0, probability=0.5),
            ProbBucket(label="up", lower=0.0, upper=None, probability=0.5),
        ],
        expected_return=0.0, upside_prob=0.5, downside_prob=0.5, var_95=-0.1,
    )
    assert _primary_large_move([fc]) is None
