"""Price-series data-quality checks."""

from __future__ import annotations

from datetime import date, timedelta

from stock_agent.data.validation import has_errors, validate_price_series
from stock_agent.schemas.market import PriceBar, PriceSeries


def _series(dates: list[date]) -> PriceSeries:
    bars = [PriceBar(date=d, open=1.0, high=2.0, low=0.5, close=1.5) for d in dates]
    return PriceSeries(ticker="X", bars=bars)


def test_clean_series_has_no_issues() -> None:
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(30)]
    issues = validate_price_series(_series(dates), min_bars=20, today=date(2025, 1, 30))
    assert issues == []


def test_short_history_flagged() -> None:
    issues = validate_price_series(
        _series([date(2025, 1, 2), date(2025, 1, 3)]), today=date(2025, 1, 3)
    )
    assert any(i.code == "short_history" for i in issues)


def test_date_gap_flagged() -> None:
    issues = validate_price_series(
        _series([date(2025, 1, 1), date(2025, 1, 20)]), min_bars=1, today=date(2025, 1, 20)
    )
    assert any(i.code == "date_gap" for i in issues)


def test_stale_data_flagged() -> None:
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(25)]
    issues = validate_price_series(_series(dates), today=date(2025, 3, 1))
    assert any(i.code == "stale_data" for i in issues)


def test_future_date_is_error() -> None:
    issues = validate_price_series(
        _series([date(2025, 1, 1), date(2025, 1, 2)]), min_bars=1, today=date(2024, 1, 1)
    )
    assert has_errors(issues)
