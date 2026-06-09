"""Conformance for the multi-ticker comparison schemas (Enhancement B)."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from stock_agent.schemas.comparison import (
    ForecastComparison,
    HeadlineRef,
    NewsComparison,
    TickerForecastRow,
    TickerNewsRow,
)


def test_forecast_comparison_roundtrips_json() -> None:
    comp = ForecastComparison(
        horizon_days=20,
        model="ensemble",
        rows=[
            TickerForecastRow(
                ticker="NVDA",
                horizon_days=20,
                model_name="ensemble",
                expected_return=0.031,
                upside_prob=0.58,
                downside_prob=0.42,
                var_95=-0.12,
                ci_low=-0.18,
                ci_high=0.24,
                prob_large_move=0.33,
                large_move_threshold_pct=5,
            ),
            TickerForecastRow(ticker="BADX", error="no data"),
        ],
        skipped=["EXTRA"],
    )
    dumped = comp.model_dump(mode="json")
    assert ForecastComparison.model_validate(dumped) == comp
    # An errored row carries no numbers; a good row does.
    assert dumped["rows"][1]["error"] == "no data"
    assert dumped["rows"][0]["upside_prob"] == 0.58


def test_probabilities_are_bounded() -> None:
    with pytest.raises(ValidationError):
        TickerForecastRow(ticker="NVDA", upside_prob=1.5)
    with pytest.raises(ValidationError):
        TickerNewsRow(ticker="NVDA", pct_positive=-0.1)


def test_news_comparison_with_headlines() -> None:
    comp = NewsComparison(
        days=14,
        rows=[
            TickerNewsRow(
                ticker="MSFT",
                article_count=12,
                avg_sentiment=0.18,
                pct_positive=0.5,
                pct_negative=0.17,
                sentiment_source="alpha_vantage",
                top_headlines=[
                    HeadlineRef(
                        title="MSFT ships chips",
                        source="Reuters",
                        published=date(2026, 6, 1),
                        url="https://x.com/a",
                    )
                ],
            )
        ],
    )
    dumped = comp.model_dump(mode="json")
    assert NewsComparison.model_validate(dumped) == comp
    assert dumped["rows"][0]["top_headlines"][0]["published"] == "2026-06-01"


def test_horizon_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        ForecastComparison(horizon_days=0, model="ensemble", rows=[])
