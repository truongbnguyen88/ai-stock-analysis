"""Analyze pipeline: prices + indicators + forecasts + news -> ResearchReport.

Thin orchestration over the data/indicator/forecast/news/report modules. Degrades
gracefully: if the LLM is disabled or fails, the report is still produced (with an
uncertainty note); horizons without enough history are skipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from stock_agent.data.earnings import fetch_earnings_context
from stock_agent.data.loader import PriceLoader
from stock_agent.features.news_features import build_news_features
from stock_agent.forecasting.base import ForecastModel
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.forecasting.ml import MLForecaster
from stock_agent.forecasting.pooled import default_model_path
from stock_agent.indicators.snapshot import compute_snapshot
from stock_agent.llm.client import AnthropicClient, LLMError, TextLLM
from stock_agent.llm.news_summarizer import SummaryGuardError, summarize_news
from stock_agent.llm.synthesizer import SynthesisGuardError, synthesize
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

    # Baseline forecasts (skip horizons lacking enough history). The report stays a
    # transparent multi-model view (baseline + ML overlay); the ensemble is the default
    # on the interactive surfaces (agent + `forecast` CLI), available here via --model.
    model: ForecastModel = HistoricalSimulation()
    forecasts = []
    for h in horizons:
        try:
            forecasts.append(model.forecast(series, horizon_days=h, as_of=as_of))
        except ValueError:
            log.info("analyze.skip_horizon", ticker=ticker, horizon=h, bars=len(series))

    # Calibrated ML overlay (lightgbm) where a trained artifact exists — its niche is
    # big-move magnitude, shown alongside the historical baseline. Skipped silently
    # for horizons without an artifact (e.g. the dropped h5), so analyze never blocks
    # on model availability and the report degrades to baselines-only.
    ml = MLForecaster("lightgbm", registry=registry)
    models_dir = Path(settings.output_dir) / "models"
    for h in horizons:
        if not default_model_path(models_dir, "lightgbm", h).exists():
            continue
        try:
            forecasts.append(ml.forecast(series, horizon_days=h, as_of=as_of))
        except (ValueError, KeyError) as exc:
            log.info("analyze.ml_skip", ticker=ticker, horizon=h, error=str(exc))

    # News (always fetched; summary optional).
    news_bundle = NewsFetcher(registry).fetch(
        ticker, lookback_days=days, company_name=company_name, top_n=25
    )

    # Earnings proximity context (flag if earnings fall within the LONGEST horizon).
    earnings = fetch_earnings_context(
        registry, ticker, as_of=as_of, horizon_days=max(horizons) if horizons else 20
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

    # Aggregate news sentiment (free AV scores) — context for the synthesis.
    news_sentiment = build_news_features(news_bundle)

    # Integrated analysis (Role C): reconcile forecast with news/earnings/technicals.
    synthesis = None
    if client is not None and use_llm and forecasts:
        try:
            synthesis = synthesize(
                ticker,
                forecasts=forecasts,
                snapshot=snapshot,
                llm=client,
                news_summary=summary,
                news_sentiment=news_sentiment,
                earnings=earnings,
            )
        except (LLMError, SynthesisGuardError, ValueError, ValidationError) as exc:
            log.warning("analyze.synthesis_failed", ticker=ticker, error=str(exc))

    return build_report(
        ticker=ticker,
        as_of=as_of,
        snapshot=snapshot,
        forecasts=forecasts,
        news_bundle=news_bundle,
        news_summary=summary,
        data_issue_messages=[f"{i.code}: {i.message}" for i in load.issues],
        n_price_bars=len(series),
        earnings=earnings,
        synthesis=synthesis,
    )
