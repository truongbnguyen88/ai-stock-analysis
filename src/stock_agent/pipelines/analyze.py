"""Analyze pipeline: prices + indicators + forecasts + news -> ResearchReport.

Thin orchestration over the data/indicator/forecast/news/report modules. Degrades
gracefully: if the LLM is disabled or fails, the report is still produced (with an
uncertainty note); horizons without enough history are skipped.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from stock_agent.data.loader import PriceLoader
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.indicators.snapshot import compute_snapshot
from stock_agent.llm.client import AnthropicClient, LLMError, TextLLM
from stock_agent.llm.news_summarizer import SummaryGuardError, summarize_news
from stock_agent.logging_config import get_logger
from stock_agent.news.fetch import NewsFetcher
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.reports.builder import build_report
from stock_agent.schemas.report import ResearchReport
from stock_agent.settings import Settings

log = get_logger(__name__)

DEFAULT_HORIZONS: tuple[int, ...] = (5, 20, 60)
# Prices fetched over a wide window so MA200 and forecast samples exist, even when
# the user's news window (`days`) is short.
_PRICE_LOOKBACK_DAYS = 420


def run_analyze(
    ticker: str,
    *,
    days: int = 30,
    settings: Settings,
    registry: ProviderRegistry | None = None,
    llm: TextLLM | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    company_name: str | None = None,
    use_llm: bool = True,
) -> ResearchReport:
    """Run the full analyze pipeline and return the assembled report."""
    registry = registry or build_default_registry(settings)
    ticker = ticker.upper()

    # Prices + indicators.
    load = PriceLoader(registry).load_recent(ticker, max(days, _PRICE_LOOKBACK_DAYS), min_bars=30)
    series = load.series
    snapshot = compute_snapshot(series)
    as_of = series.bars[-1].date

    # Baseline forecasts (skip horizons lacking enough history).
    model = HistoricalSimulation()
    forecasts = []
    for h in horizons:
        try:
            forecasts.append(model.forecast(series, horizon_days=h, as_of=as_of))
        except ValueError:
            log.info("analyze.skip_horizon", ticker=ticker, horizon=h, bars=len(series))

    # News (always fetched; summary optional).
    news_bundle = NewsFetcher(registry).fetch(
        ticker, lookback_days=days, company_name=company_name, top_n=25
    )

    summary = None
    client = llm
    if client is None and use_llm and settings.anthropic_api_key:
        client = AnthropicClient(settings)
    if client is not None and use_llm:
        try:
            summary = summarize_news(news_bundle, client)
        except (LLMError, SummaryGuardError, ValueError, ValidationError) as exc:
            # Degrade gracefully: a failed/garbled summary must not sink the report.
            log.warning("analyze.summary_failed", ticker=ticker, error=str(exc))

    return build_report(
        ticker=ticker,
        as_of=as_of,
        snapshot=snapshot,
        forecasts=forecasts,
        news_bundle=news_bundle,
        news_summary=summary,
        data_issue_messages=[f"{i.code}: {i.message}" for i in load.issues],
        n_price_bars=len(series),
    )
