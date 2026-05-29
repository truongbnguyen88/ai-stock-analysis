"""End-to-end analyze pipeline over fakes (no network, no live LLM)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

from stock_agent.pipelines.analyze import run_analyze
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.reports.render_md import render_markdown
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings


class FakeLLM:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        return self._payload


def _series() -> PriceSeries:
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i),
            open=100.0 + i,
            high=100.0 + i + 0.5,
            low=100.0 + i - 0.5,
            close=100.0 + i,
        )
        for i in range(60)
    ]
    return PriceSeries(ticker="NVDA", bars=bars)


def _news() -> NewsBundle:
    art = Article(
        title="NVDA launches new chip",
        url="https://x.com/a",
        source="finnhub",
        published_at=datetime.now(UTC) - timedelta(days=2),
        summary="big launch",
    )
    return NewsBundle(ticker="NVDA", articles=[art])


def _summary_json() -> str:
    return json.dumps(
        {
            "overview": "NVDA had a strong month.",
            "key_themes": ["AI demand"],
            "bullish": [{"point": "New chip launch", "sources": ["https://x.com/a"]}],
            "bearish": [],
            "risks": [],
            "catalysts": [{"point": "Upcoming earnings", "sources": []}],
        }
    )


def test_run_analyze_end_to_end() -> None:
    fake = FakeProvider("fake", prices=_series(), news=_news())
    registry = ProviderRegistry(
        [fake],
        Settings(_env_file=None, provider_price_priority="fake", provider_news_priority="fake"),
    )

    report = run_analyze(
        "nvda",
        days=30,
        settings=Settings(_env_file=None),
        registry=registry,
        llm=FakeLLM(_summary_json()),
        horizons=(5, 20),
        use_llm=True,
    )

    assert report.ticker == "NVDA"
    assert len(report.forecasts) == 2
    assert "New chip launch" in report.bullish_factors  # from the (fake) LLM summary
    assert report.news_analysis.positive_themes == ["New chip launch"]

    md = render_markdown(report)
    assert "# Research Report — NVDA" in md
    assert "## Probabilistic Scenarios" in md


def test_run_analyze_without_llm() -> None:
    fake = FakeProvider("fake", prices=_series(), news=_news())
    registry = ProviderRegistry(
        [fake],
        Settings(_env_file=None, provider_price_priority="fake", provider_news_priority="fake"),
    )
    report = run_analyze(
        "NVDA", days=30, settings=Settings(_env_file=None), registry=registry, use_llm=False
    )
    # No summary -> raw headline used; baseline forecasts still present.
    assert report.news_analysis.recent_developments == ["NVDA launches new chip"]
    assert len(report.forecasts) == 2
