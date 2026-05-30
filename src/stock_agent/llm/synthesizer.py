"""Role C: integrative synthesis of forecast + qualitative signals.

Reconciles the quantitative forecast(s) with the news summary, sentiment, earnings
timing, and technicals into an "Integrated Analysis". The LLM interprets the
numbers but may not invent or revise them — enforced by the numeric-grounding
guard (every figure in the output must trace to an input signal). One corrective
retry, then it raises rather than emit ungrounded figures.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from stock_agent.indicators.snapshot import IndicatorSnapshot
from stock_agent.llm.client import TextLLM
from stock_agent.llm.guards import NewsSummary, NumberGrounding
from stock_agent.llm.prompts.synthesis import SYSTEM, build_user
from stock_agent.logging_config import get_logger
from stock_agent.schemas.earnings import EarningsContext
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.synthesis import Synthesis

log = get_logger(__name__)


class SynthesisGuardError(RuntimeError):
    """Raised when the synthesis keeps stating ungrounded figures after a retry."""


def _loads_lenient(text: str) -> dict[str, Any]:
    try:
        result: Any = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("synthesis output was not a JSON object")
    return result


def _synthesis_text(s: Synthesis) -> str:
    return " ".join([s.overview, *s.alignments, *s.tensions, s.confidence])


def _build_signals(
    forecasts: Sequence[ScenarioForecast],
    snapshot: IndicatorSnapshot,
    news_summary: NewsSummary | None,
    news_sentiment: dict[str, float] | None,
    earnings: EarningsContext | None,
    grounding: NumberGrounding,
) -> list[str]:
    """Render signal lines for the prompt AND feed the grounding set from the same data."""
    lines: list[str] = []

    for fc in forecasts:
        grounding.add_from(fc.model_dump())
        var = f", VaR95={fc.var_95:.1%}" if fc.var_95 is not None else ""
        lines.append(
            f"- Forecast [{fc.model_name}, {fc.horizon_days}d]: P(up)={fc.upside_prob:.0%}, "
            f"P(down)={fc.downside_prob:.0%}, E[r]={fc.expected_return:+.1%}, "
            f"P(>+10%)={fc.buckets[-1].probability:.0%}{var}"
        )

    grounding.add_from(snapshot.numeric_indicators())
    tech = f"- Technicals: trend={snapshot.trend_label}"
    if snapshot.rsi14 is not None:
        tech += f", RSI={snapshot.rsi14:.0f}"
    if snapshot.hist_vol_annualized is not None:
        tech += f", annualized vol={snapshot.hist_vol_annualized:.0%}"
    if snapshot.macd_hist is not None:
        tech += f", MACD histogram {'positive' if snapshot.macd_hist >= 0 else 'negative'}"
    lines.append(tech)

    if news_sentiment:
        grounding.add_from(news_sentiment)
        lines.append(
            f"- News sentiment: avg={news_sentiment.get('avg_sentiment', 0.0):.2f} "
            f"(coverage {news_sentiment.get('sentiment_coverage', 0.0):.0%}), "
            f"%positive={news_sentiment.get('pct_positive', 0.0):.0%}, "
            f"%negative={news_sentiment.get('pct_negative', 0.0):.0%}"
        )

    if news_summary is not None:
        grounding.add_from(news_summary.model_dump())
        if news_summary.key_themes:
            lines.append(f"- News themes: {'; '.join(news_summary.key_themes[:5])}")
        if news_summary.bullish:
            lines.append(f"- News bullish: {'; '.join(p.point for p in news_summary.bullish[:3])}")
        if news_summary.bearish:
            lines.append(f"- News bearish: {'; '.join(p.point for p in news_summary.bearish[:3])}")
        if news_summary.risks:
            lines.append(f"- News risks: {'; '.join(p.point for p in news_summary.risks[:3])}")

    if earnings is not None and earnings.next_earnings_date is not None:
        grounding.add_from(earnings.model_dump())
        loc = "INSIDE" if earnings.earnings_in_horizon else "outside"
        lines.append(
            f"- Earnings: next {earnings.next_earnings_date} in "
            f"{earnings.days_to_next_earnings} days ({loc} the {earnings.horizon_days}d horizon)"
        )

    return lines


def synthesize(
    ticker: str,
    *,
    forecasts: Sequence[ScenarioForecast],
    snapshot: IndicatorSnapshot,
    llm: TextLLM,
    news_summary: NewsSummary | None = None,
    news_sentiment: dict[str, float] | None = None,
    earnings: EarningsContext | None = None,
) -> Synthesis:
    """Produce a grounded integrated analysis reconciling forecast with qualitative signals."""
    grounding = NumberGrounding()
    signal_lines = _build_signals(
        forecasts, snapshot, news_summary, news_sentiment, earnings, grounding
    )
    user = build_user(ticker, signal_lines)

    raw = llm.complete_json(system=SYSTEM, user=user, max_tokens=1500)
    result = Synthesis.model_validate(_loads_lenient(raw))

    violations = grounding.ungrounded(_synthesis_text(result))
    if violations:
        log.warning("synthesis.ungrounded", ticker=ticker, figures=violations[:5])
        correction = (
            user
            + "\n\nIMPORTANT: these figures in your answer are not in the inputs: "
            + ", ".join(violations[:5])
            + ". Restate using ONLY the numbers given above, or remove them. Do not invent "
            "or revise any probability or return."
        )
        result = Synthesis.model_validate(
            _loads_lenient(llm.complete_json(system=SYSTEM, user=correction, max_tokens=1500))
        )
        remaining = grounding.ungrounded(_synthesis_text(result))
        if remaining:
            raise SynthesisGuardError(f"ungrounded figures persisted: {', '.join(remaining[:5])}")

    return result
