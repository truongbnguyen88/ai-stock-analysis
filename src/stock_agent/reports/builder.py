"""Assemble a ``ResearchReport`` from structured inputs.

Deterministic composition: narrative text is templated directly from the computed
numbers and the (LLM-authored) news summary. The LLM does not write the report —
it only supplied the news synthesis upstream. No probabilities or recommendations
are invented here; quantitative fields come from the indicator snapshot and the
forecast models.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date

from stock_agent.indicators.snapshot import IndicatorSnapshot
from stock_agent.llm.guards import NewsSummary
from stock_agent.news.clean import canonical_url
from stock_agent.schemas.earnings import EarningsContext
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import Fundamentals
from stock_agent.schemas.news import NewsBundle
from stock_agent.schemas.report import (
    Citation,
    NewsAnalysisSection,
    ResearchReport,
    TechnicalAnalysisSection,
)

_MAX_CITATIONS = 12


def _rsi_state(rsi: float | None) -> str | None:
    if rsi is None:
        return None
    if rsi >= 70:
        return f"RSI {rsi:.0f} (overbought territory)"
    if rsi <= 30:
        return f"RSI {rsi:.0f} (oversold territory)"
    return f"RSI {rsi:.0f} (neutral)"


def _technical_narrative(snap: IndicatorSnapshot) -> str:
    """Plain-language description of the indicator snapshot (numbers only)."""
    parts = [f"Last close {snap.last_close:,.2f}."]
    if snap.trend_label != "insufficient_data":
        loc = "above" if snap.price_above_ma50 else "below"
        parts.append(f"Trend regime is {snap.trend_label} (price {loc} MA50).")
    state = _rsi_state(snap.rsi14)
    if state:
        parts.append(state + ".")
    if snap.hist_vol_annualized is not None:
        parts.append(f"Annualized volatility ~{snap.hist_vol_annualized:.0%}.")
    if snap.macd_hist is not None:
        mom = "positive" if snap.macd_hist >= 0 else "negative"
        parts.append(f"MACD histogram is {mom}.")
    parts.append(f"Max drawdown over the window {snap.max_drawdown:.0%}.")
    return " ".join(parts)


def _technical_flags(snap: IndicatorSnapshot) -> tuple[list[str], list[str], list[str]]:
    """Descriptive (not advisory) technical bullish/bearish/risk flags."""
    bullish: list[str] = []
    bearish: list[str] = []
    risks: list[str] = []
    if snap.ma50_above_ma200 and snap.price_above_ma50:
        bullish.append("Price in an uptrend: above MA50 with MA50 above MA200.")
    if snap.ma50_above_ma200 is False and snap.price_above_ma50 is False:
        bearish.append("Price in a downtrend: below MA50 with MA50 below MA200.")
    if snap.rsi14 is not None and snap.rsi14 >= 70:
        risks.append(f"RSI {snap.rsi14:.0f}: overbought, elevated pullback risk.")
    if snap.rsi14 is not None and snap.rsi14 <= 30:
        risks.append(f"RSI {snap.rsi14:.0f}: oversold conditions.")
    if snap.hist_vol_annualized is not None and snap.hist_vol_annualized > 0.5:
        risks.append(f"Elevated annualized volatility (~{snap.hist_vol_annualized:.0%}).")
    if snap.max_drawdown < -0.30:
        risks.append(f"Deep drawdown from peak ({snap.max_drawdown:.0%}).")
    return bullish, bearish, risks


def _news_section(summary: NewsSummary | None, bundle: NewsBundle) -> NewsAnalysisSection:
    if summary is None:
        # LLM disabled: fall back to raw headlines as recent developments.
        return NewsAnalysisSection(
            recent_developments=[a.title for a in bundle.articles[:5]],
        )
    return NewsAnalysisSection(
        recent_developments=list(summary.key_themes),
        positive_themes=[p.point for p in summary.bullish],
        negative_themes=[p.point for p in summary.bearish],
        emerging_risks=[p.point for p in summary.risks],
    )


def _citations(summary: NewsSummary | None, bundle: NewsBundle) -> list[Citation]:
    """Cite the articles actually referenced by the summary, else the top articles."""
    cited_canonical: set[str] = set()
    if summary is not None:
        for group in (summary.bullish, summary.bearish, summary.risks, summary.catalysts):
            for point in group:
                cited_canonical.update(canonical_url(u) for u in point.sources)

    chosen = [a for a in bundle.articles if canonical_url(str(a.url)) in cited_canonical]
    if not chosen:
        chosen = list(bundle.articles[:5])  # fallback: most relevant articles
    return [Citation(title=a.title, url=a.url, source=a.source) for a in chosen[:_MAX_CITATIONS]]


def _uncertainty_notes(
    snap: IndicatorSnapshot,
    forecasts: Sequence[ScenarioForecast],
    bundle: NewsBundle,
    data_issue_messages: Sequence[str],
    *,
    has_summary: bool,
) -> list[str]:
    notes: list[str] = list(data_issue_messages)
    if len(bundle.articles) == 0:
        notes.append("No recent news found for the window.")
    elif len(bundle.articles) < 5:
        notes.append(f"Sparse news coverage ({len(bundle.articles)} articles).")
    if not has_summary:
        notes.append(
            "News summary unavailable (LLM disabled or failed); themes/signals not extracted."
        )
    if snap.ma200 is None:
        notes.append("Limited price history: 200-day moving average unavailable.")
    for fc in forecasts:
        if fc.notes:
            notes.append(f"{fc.horizon_days}d forecast: {fc.notes}")
    notes.append(
        "Forecast probabilities use a historical-simulation baseline and are uncalibrated "
        "(calibration arrives in a later phase)."
    )
    return notes


def _pick_headline_forecast(forecasts: Sequence[ScenarioForecast]) -> ScenarioForecast | None:
    """Prefer a ~20-day horizon for the executive highlight."""
    if not forecasts:
        return None
    return min(forecasts, key=lambda f: abs(f.horizon_days - 20))


def build_report(
    *,
    ticker: str,
    as_of: Date,
    snapshot: IndicatorSnapshot,
    forecasts: Sequence[ScenarioForecast],
    news_bundle: NewsBundle,
    news_summary: NewsSummary | None = None,
    data_issue_messages: Sequence[str] = (),
    n_price_bars: int = 0,
    fundamentals: Fundamentals | None = None,
    earnings: EarningsContext | None = None,
) -> ResearchReport:
    """Compose the full research report from computed inputs."""
    tech_bull, tech_bear, tech_risk = _technical_flags(snapshot)

    bullish = ([p.point for p in news_summary.bullish] if news_summary else []) + tech_bull
    bearish = ([p.point for p in news_summary.bearish] if news_summary else []) + tech_bear
    risks = ([p.point for p in news_summary.risks] if news_summary else []) + tech_risk

    # Executive summary (templated; describes numbers, recommends nothing).
    exec_parts = [f"{ticker}: technical trend is {snapshot.trend_label}"]
    if snapshot.hist_vol_annualized is not None:
        exec_parts.append(f" with annualized volatility ~{snapshot.hist_vol_annualized:.0%}")
    exec_parts.append(".")
    if news_summary and news_summary.overview:
        exec_parts.append(" " + news_summary.overview)
    headline = _pick_headline_forecast(forecasts)
    if headline:
        exec_parts.append(
            f" Over {headline.horizon_days} trading days, the baseline model estimates a "
            f"{headline.upside_prob:.0%} chance of a positive return "
            f"(expected {headline.expected_return:+.1%})."
        )
    executive_summary = "".join(exec_parts)

    # Key observations.
    observations: list[str] = []
    if snapshot.trend_label != "insufficient_data":
        observations.append(f"Trend regime: {snapshot.trend_label}.")
    state = _rsi_state(snapshot.rsi14)
    if state:
        observations.append(state + ".")
    if snapshot.hist_vol_annualized is not None:
        observations.append(f"Annualized volatility ~{snapshot.hist_vol_annualized:.0%}.")
    if news_summary and news_summary.key_themes:
        observations.append(f"Top news theme: {news_summary.key_themes[0]}.")
    for fc in forecasts:
        observations.append(
            f"{fc.horizon_days}d: P(up)={fc.upside_prob:.0%}, "
            f"E[r]={fc.expected_return:+.1%}, P(>+10%)={fc.buckets[-1].probability:.0%}."
        )
    if news_summary:
        for cat in news_summary.catalysts[:2]:
            observations.append(f"Catalyst: {cat.point}")
    if earnings is not None and earnings.next_earnings_date is not None:
        observations.append(
            f"Next earnings {earnings.next_earnings_date} ({earnings.days_to_next_earnings}d away)."
        )

    confidence_notes = (
        f"Based on {len(news_bundle.articles)} news articles and {n_price_bars} price bars. "
        "Forecasts are a historical-simulation baseline (uncalibrated)."
    )

    uncertainty = list(
        _uncertainty_notes(
            snapshot,
            forecasts,
            news_bundle,
            data_issue_messages,
            has_summary=news_summary is not None,
        )
    )
    # The single most useful earnings caveat: a scheduled event the price-only
    # forecast cannot see, which widens the actual return distribution.
    if earnings is not None and earnings.earnings_in_horizon:
        uncertainty.append(
            f"Earnings on {earnings.next_earnings_date} fall WITHIN the forecast horizon — "
            "the model is price-only and cannot price this event; expect a wider, "
            "more bimodal outcome than the forecast suggests."
        )

    return ResearchReport(
        ticker=ticker,
        as_of=as_of,
        executive_summary=executive_summary,
        key_observations=observations,
        confidence_notes=confidence_notes,
        technical_analysis=TechnicalAnalysisSection(
            narrative=_technical_narrative(snapshot),
            indicators=snapshot.numeric_indicators(),
        ),
        news_analysis=_news_section(news_summary, news_bundle),
        fundamentals=fundamentals,
        earnings=earnings,
        forecasts=list(forecasts),
        bullish_factors=bullish,
        bearish_factors=bearish,
        risk_flags=risks,
        uncertainty_notes=uncertainty,
        citations=_citations(news_summary, news_bundle),
    )
