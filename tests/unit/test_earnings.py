"""Tests for earnings context (display) and the leakage-safe cadence estimate."""

from __future__ import annotations

from datetime import date

from stock_agent.data.earnings import (
    days_to_next_earnings_estimate,
    earnings_cadence_days,
    earnings_context,
)


def test_context_next_and_proximity() -> None:
    dates = [date(2025, 1, 15), date(2025, 4, 15), date(2025, 7, 15)]
    ctx = earnings_context(dates, ticker="x", as_of=date(2025, 4, 20), horizon_days=20)
    assert ctx.last_earnings_date == date(2025, 4, 15)
    assert ctx.next_earnings_date == date(2025, 7, 15)
    assert ctx.days_since_last_earnings == 5
    assert ctx.days_to_next_earnings == (date(2025, 7, 15) - date(2025, 4, 20)).days
    assert ctx.earnings_in_horizon is False  # ~86 days > 20


def test_context_in_horizon() -> None:
    ctx = earnings_context(
        [date(2025, 4, 15), date(2025, 5, 5)], ticker="x", as_of=date(2025, 4, 25), horizon_days=20
    )
    assert ctx.days_to_next_earnings == 10
    assert ctx.earnings_in_horizon is True


def test_cadence_quarterly_and_default() -> None:
    assert 85 <= earnings_cadence_days([date(2025, 1, 1), date(2025, 4, 1), date(2025, 7, 1)]) <= 95
    assert earnings_cadence_days([date(2025, 1, 1)]) == 91.0  # too few → default


def test_estimate_uses_only_past_dates() -> None:
    # A future date in the list must be ignored (leakage guard).
    past_and_future = [date(2025, 1, 15), date(2025, 4, 15), date(2025, 12, 1)]
    est = days_to_next_earnings_estimate(date(2025, 5, 15), past_and_future, cadence=91.0)
    assert est == 61.0  # 91 - (days since 2025-04-15 = 30)


def test_estimate_clamps_to_zero_when_overdue() -> None:
    est = days_to_next_earnings_estimate(date(2025, 8, 1), [date(2025, 4, 15)], cadence=91.0)
    assert est == 0.0  # days_since 108 > cadence → imminent


def test_estimate_none_without_prior_earnings() -> None:
    est = days_to_next_earnings_estimate(date(2025, 1, 1), [date(2025, 6, 1)], cadence=91.0)
    assert est is None
