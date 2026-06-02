"""Tests for report assembly and markdown rendering."""

from __future__ import annotations

from datetime import UTC, date, datetime

from stock_agent.forecasting.historical import historical_forecast
from stock_agent.indicators.snapshot import IndicatorSnapshot
from stock_agent.llm.guards import CitedPoint, NewsSummary
from stock_agent.reports.builder import build_report
from stock_agent.reports.render_md import render_markdown
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.news import Article, NewsBundle


def _snapshot() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        as_of=date(2025, 1, 31),
        last_close=214.25,
        last_daily_return=0.012,
        ma20=214.88,
        ma50=198.73,
        ma200=187.5,
        rsi14=52.6,
        macd=4.6,
        macd_signal=6.5,
        macd_hist=-1.9,
        hist_vol_annualized=0.40,
        atr14=7.17,
        max_drawdown=-0.20,
        price_above_ma50=True,
        price_above_ma200=True,
        ma50_above_ma200=True,
        trend_label="uptrend",
    )


def _bundle() -> NewsBundle:
    art = Article(
        title="NVDA launches new chip",
        url="https://x.com/a",
        source="finnhub",
        published_at=datetime(2025, 1, 30, tzinfo=UTC),
        summary="launch",
    )
    return NewsBundle(ticker="NVDA", articles=[art])


def _summary() -> NewsSummary:
    return NewsSummary(
        overview="NVDA had a strong month.",
        key_themes=["AI demand"],
        bullish=[CitedPoint(point="New chip launch", sources=["https://x.com/a"])],
        bearish=[CitedPoint(point="Valuation concerns")],
        risks=[CitedPoint(point="Regulatory scrutiny")],
        catalysts=[CitedPoint(point="Upcoming earnings")],
    )


def _forecasts() -> list[ScenarioForecast]:
    closes = [100.0 * (1.01**i) for i in range(60)]
    return [historical_forecast(closes, h, ticker="NVDA", as_of=date(2025, 1, 31)) for h in (5, 20)]


def test_build_report_merges_news_and_technical_factors() -> None:
    report = build_report(
        ticker="NVDA",
        as_of=date(2025, 1, 31),
        snapshot=_snapshot(),
        forecasts=_forecasts(),
        news_bundle=_bundle(),
        news_summary=_summary(),
        data_issue_messages=["stale_data: latest bar is 6 days old"],
        n_price_bars=60,
    )
    # news bullish point + technical uptrend flag
    assert "New chip launch" in report.bullish_factors
    assert any("uptrend" in b.lower() for b in report.bullish_factors)
    assert "Valuation concerns" in report.bearish_factors
    assert "Regulatory scrutiny" in report.risk_flags
    # citation resolved from cited URL
    assert any(str(c.url).startswith("https://x.com/a") for c in report.citations)
    # uncertainty carries the data issue + baseline caveat
    assert any("stale_data" in n for n in report.uncertainty_notes)
    assert any("uncalibrated" in n for n in report.uncertainty_notes)


def test_build_report_without_llm_summary() -> None:
    report = build_report(
        ticker="NVDA",
        as_of=date(2025, 1, 31),
        snapshot=_snapshot(),
        forecasts=_forecasts(),
        news_bundle=_bundle(),
        news_summary=None,
        n_price_bars=60,
    )
    assert report.news_analysis.recent_developments == ["NVDA launches new chip"]
    assert any("LLM disabled" in n for n in report.uncertainty_notes)


def test_render_markdown_has_all_sections_and_disclaimer() -> None:
    report = build_report(
        ticker="NVDA",
        as_of=date(2025, 1, 31),
        snapshot=_snapshot(),
        forecasts=_forecasts(),
        news_bundle=_bundle(),
        news_summary=_summary(),
        n_price_bars=60,
    )
    md = render_markdown(report)
    for header in (
        "# Research Report — NVDA",
        "Not financial advice",
        "## Executive Summary",
        "## Technical Analysis",
        "## Probabilistic Scenarios",
        "## News Analysis",
        "## Bullish Factors",
        "## Bearish Factors",
        "## Risk Flags",
        "## Uncertainty Notes",
        "## Source Citations",
    ):
        assert header in md
    assert "> +10%" in md  # a scenario bucket label
    assert "Expected return" in md


def test_forecast_block_surfaces_big_move_calibration_and_trust() -> None:
    from stock_agent.reports.render_md import _forecast_block

    # A calibrated ML forecast at h20 (±5/±10 buckets). Render must surface the
    # horizon-scaled big-move reading, the real calibration status, and a trust note.
    closes = [100.0 * (1.01**i) for i in range(80)]
    base = historical_forecast(closes, 20, ticker="NVDA", as_of=date(2025, 1, 31))
    fc = base.model_copy(update={"model_name": "ml_lightgbm", "calibration_status": "calibrated"})
    block = "\n".join(_forecast_block(fc))
    assert "Big move (|r| > 5%):" in block  # inner k for h20
    assert "leans" in block
    assert "Calibration: calibrated" in block
    assert "Trust:" in block  # h20 → measurable-skill note


def test_forecast_block_long_horizon_low_confidence_trust() -> None:
    from stock_agent.reports.render_md import _forecast_block

    closes = [100.0 * (1.004**i) for i in range(200)]
    fc = historical_forecast(closes, 60, ticker="NVDA", as_of=date(2025, 1, 31))
    block = "\n".join(_forecast_block(fc))
    assert "low-confidence" in block  # h60 → long-horizon caveat


def test_forecast_block_mid_horizon_has_no_trust_note() -> None:
    from stock_agent.reports.render_md import _forecast_block

    # 31–59 trading days is between the measurable-skill and low-confidence bands →
    # no trust claim is rendered (avoids over-stating confidence).
    closes = [100.0 * (1.005**i) for i in range(160)]
    fc = historical_forecast(closes, 45, ticker="NVDA", as_of=date(2025, 1, 31))
    assert "Trust:" not in "\n".join(_forecast_block(fc))
