"""Tests for the invariant guards: anti-forecast + citation validity."""

from __future__ import annotations

from stock_agent.llm.guards import (
    CitedPoint,
    NewsSummary,
    find_forecast_violations,
    find_invalid_citations,
    sanitize_citations,
)


def test_forecast_language_is_flagged() -> None:
    bad = NewsSummary(
        overview="There is a 70% chance the stock rallies next quarter.",
        bullish=[CitedPoint(point="I set a price target of $250.")],
    )
    violations = find_forecast_violations(bad)
    assert len(violations) >= 2


def test_reported_facts_not_flagged() -> None:
    # Past/reported figures with no forecast cue must pass.
    ok = NewsSummary(
        overview="Shares fell 5% after the company reported revenue up 20% year over year.",
        key_themes=["earnings", "guidance raised"],
    )
    assert find_forecast_violations(ok) == []


def test_reported_analyst_price_target_not_flagged() -> None:
    # Regression: "Truist raised its price target" is a news fact, not LLM fabrication.
    ok = NewsSummary(
        overview=(
            "NVIDIA reported 85.2% year-over-year revenue growth to $81.62 billion "
            "with EPS of $1.87, beating estimates."
        ),
        bullish=[
            CitedPoint(
                point="Truist Securities raised its price target and maintained a Buy rating."
            ),
            CitedPoint(point="Analyst upgraded price target to $200 citing strong Q1 results."),
        ],
    )
    assert find_forecast_violations(ok) == []


def test_llm_own_price_target_still_flagged() -> None:
    # The LLM asserting its own price target must still be caught.
    bad = NewsSummary(bullish=[CitedPoint(point="My price target is $300 for the next quarter.")])
    assert len(find_forecast_violations(bad)) >= 1


def test_invalid_citations_detected_and_sanitized() -> None:
    allowed = {"https://real.com/a"}
    summary = NewsSummary(
        bullish=[
            CitedPoint(
                point="New product launch",
                sources=["https://real.com/a", "https://fake.com/made-up"],
            )
        ]
    )
    assert find_invalid_citations(summary, allowed) == ["https://fake.com/made-up"]

    cleaned = sanitize_citations(summary, allowed)
    assert cleaned.bullish[0].sources == ["https://real.com/a"]
    assert cleaned.bullish[0].point == "New product launch"  # text preserved


def test_citation_match_ignores_tracking_params() -> None:
    allowed = {"https://real.com/a"}  # canonical form
    summary = NewsSummary(
        bullish=[CitedPoint(point="x", sources=["https://real.com/a?utm_source=feed"])]
    )
    assert find_invalid_citations(summary, allowed) == []
