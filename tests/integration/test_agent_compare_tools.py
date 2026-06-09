"""Multi-ticker batch tools (Enhancement B) — compare_forecasts / compare_news.

Offline + deterministic: a FakeProvider supplies prices/news for every ticker
(the double ignores the symbol), so the batch loop, the cap, error-row handling,
and schema conformance are all exercised without any network.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from stock_agent.agent.tools import _MAX_TICKERS, ToolExecutor
from stock_agent.providers.base import ProviderUnavailable
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.comparison import ForecastComparison, NewsComparison
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings


def _prices(n_bars: int = 260) -> PriceSeries:
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
        )
        for i in range(n_bars)
    ]
    return PriceSeries(ticker="NVDA", bars=bars)


def _executor(*, news: NewsBundle | None = None) -> ToolExecutor:
    fake = FakeProvider("fake", prices=_prices(), news=news)
    registry = ProviderRegistry(
        [fake],
        Settings(_env_file=None, provider_price_priority="fake", provider_news_priority="fake"),
    )
    return ToolExecutor(Settings(_env_file=None), registry=registry)


def _news_bundle() -> NewsBundle:
    arts = [
        Article(
            title="Chipmaker earnings beat estimates",
            url="https://x.com/1",
            source="alpha_vantage",
            published_at=datetime.now(UTC) - timedelta(days=1),
            sentiment=0.4,
        ),
        Article(
            title="Sector faces regulatory probe",
            url="https://x.com/2",
            source="alpha_vantage",
            published_at=datetime.now(UTC) - timedelta(days=2),
            sentiment=-0.2,
        ),
    ]
    return NewsBundle(ticker="NVDA", articles=arts)


# ---- compare_forecasts -------------------------------------------------------
def test_compare_forecasts_returns_one_row_per_ticker_and_conforms() -> None:
    r = _executor().execute(
        "compare_forecasts",
        {"tickers": ["NVDA", "MSFT"], "horizon_days": 20, "model": "historical_sim"},
    )
    assert "error" not in r
    comp = ForecastComparison.model_validate(r)  # schema conformance
    assert [row.ticker for row in comp.rows] == ["NVDA", "MSFT"]
    for row in comp.rows:
        assert row.error is None
        assert row.upside_prob is not None  # numbers present (from the model)
        assert row.model_name == "historical_sim"


def test_compare_forecasts_enforces_cap_and_reports_skipped() -> None:
    many = [f"T{i}" for i in range(_MAX_TICKERS + 2)]
    r = _executor().execute("compare_forecasts", {"tickers": many, "model": "historical_sim"})
    comp = ForecastComparison.model_validate(r)
    assert len(comp.rows) == _MAX_TICKERS
    assert comp.skipped == many[_MAX_TICKERS:]


def test_compare_forecasts_requires_at_least_two() -> None:
    r = _executor().execute("compare_forecasts", {"tickers": ["NVDA"]})
    assert "error" in r and "at least 2" in r["error"]


def test_compare_forecasts_rejects_unknown_model() -> None:
    r = _executor().execute(
        "compare_forecasts", {"tickers": ["NVDA", "MSFT"], "model": "nope"}
    )
    assert "error" in r and "unknown model" in r["error"]


def test_compare_forecasts_accepts_comma_string_and_dedupes() -> None:
    r = _executor().execute(
        "compare_forecasts", {"tickers": "nvda, msft ,nvda", "model": "historical_sim"}
    )
    comp = ForecastComparison.model_validate(r)
    assert [row.ticker for row in comp.rows] == ["NVDA", "MSFT"]  # upper + dedup


def test_compare_forecasts_per_ticker_error_row_on_no_data() -> None:
    # A registry whose price provider always fails -> every row carries an error,
    # but the comparison still returns (one bad ticker never sinks the batch).
    failing = FakeProvider("fake", raises=ProviderUnavailable("fake", "down"))
    registry = ProviderRegistry(
        [failing], Settings(_env_file=None, provider_price_priority="fake")
    )
    ex = ToolExecutor(Settings(_env_file=None), registry=registry)
    r = ex.execute("compare_forecasts", {"tickers": ["NVDA", "MSFT"], "model": "historical_sim"})
    comp = ForecastComparison.model_validate(r)
    assert all(row.error is not None for row in comp.rows)
    assert all(row.upside_prob is None for row in comp.rows)


# ---- compare_news ------------------------------------------------------------
def test_compare_news_returns_sentiment_and_headlines() -> None:
    r = _executor(news=_news_bundle()).execute(
        "compare_news", {"tickers": ["NVDA", "MSFT"], "days": 14}
    )
    assert "error" not in r
    comp = NewsComparison.model_validate(r)
    assert [row.ticker for row in comp.rows] == ["NVDA", "MSFT"]
    for row in comp.rows:
        assert row.error is None
        assert row.article_count == 2
        assert row.sentiment_source == "alpha_vantage"
        assert row.top_headlines  # newest-first headlines present


def test_compare_news_enforces_cap() -> None:
    many = [f"T{i}" for i in range(_MAX_TICKERS + 3)]
    r = _executor(news=_news_bundle()).execute("compare_news", {"tickers": many})
    comp = NewsComparison.model_validate(r)
    assert len(comp.rows) == _MAX_TICKERS
    assert comp.skipped == many[_MAX_TICKERS:]


def test_compare_news_requires_at_least_two() -> None:
    r = _executor(news=_news_bundle()).execute("compare_news", {"tickers": ["NVDA"]})
    assert "error" in r and "at least 2" in r["error"]
