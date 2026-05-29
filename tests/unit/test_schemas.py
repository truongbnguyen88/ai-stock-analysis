"""Schema validation tests — the invariants downstream code relies on."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from stock_agent.schemas import (
    PriceBar,
    PriceSeries,
    ProbBucket,
    ResearchReport,
    ScenarioForecast,
)


def _bar(d: date, c: float) -> PriceBar:
    """Construct a valid bar centered on close ``c``."""
    return PriceBar(date=d, open=c, high=c + 1, low=c - 1, close=c, volume=100)


def test_pricebar_rejects_high_below_low() -> None:
    with pytest.raises(ValidationError):
        PriceBar(date=date(2025, 1, 1), open=10, high=5, low=8, close=9)


def test_pricebar_rejects_close_outside_range() -> None:
    with pytest.raises(ValidationError):
        PriceBar(date=date(2025, 1, 1), open=10, high=11, low=9, close=12)


def test_priceseries_requires_sorted_unique_dates() -> None:
    out_of_order = [_bar(date(2025, 1, 2), 10), _bar(date(2025, 1, 1), 11)]
    with pytest.raises(ValidationError):
        PriceSeries(ticker="X", bars=out_of_order)


def test_priceseries_closes_prefers_adjusted() -> None:
    bar = PriceBar(date=date(2025, 1, 1), open=10, high=11, low=9, close=10, adj_close=9.5)
    series = PriceSeries(ticker="X", bars=[bar])
    assert series.closes == [9.5]
    assert len(series) == 1


def test_scenarioforecast_buckets_must_sum_to_one() -> None:
    buckets = [
        ProbBucket(label="up", lower=0.0, upper=None, probability=0.5),
        ProbBucket(label="down", lower=None, upper=0.0, probability=0.3),  # sums to 0.8
    ]
    with pytest.raises(ValidationError):
        ScenarioForecast(
            ticker="X",
            as_of=date(2025, 1, 1),
            horizon_days=20,
            model_name="historical_sim",
            buckets=buckets,
            expected_return=0.0,
            upside_prob=0.5,
            downside_prob=0.5,
        )


def test_scenarioforecast_accepts_valid_distribution() -> None:
    buckets = [
        ProbBucket(label="up", lower=0.0, upper=None, probability=0.6),
        ProbBucket(label="down", lower=None, upper=0.0, probability=0.4),
    ]
    fc = ScenarioForecast(
        ticker="X",
        as_of=date(2025, 1, 1),
        horizon_days=20,
        model_name="historical_sim",
        buckets=buckets,
        expected_return=0.01,
        upside_prob=0.6,
        downside_prob=0.4,
    )
    assert fc.calibration_status == "unknown"


def test_research_report_has_no_recommendation_field() -> None:
    # Guards the non-advisory invariant: if anyone adds a recommendation field,
    # this test is the first to fail.
    assert "recommendation" not in ResearchReport.model_fields
