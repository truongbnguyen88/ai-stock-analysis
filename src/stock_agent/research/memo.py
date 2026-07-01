"""Integrated research memo (RAG P8) — the capstone synthesis + render.

`build_memo` assembles one memo for a ticker from already-gathered inputs (the pipeline does
the I/O): the QUANTITATIVE sections (`technical_indicators`, `forecasts`) are copied verbatim
from the models, and ONE grounded LLM call produces the qualitative narrative — executive
summary, management commentary, business drivers, risk factors, bullish/bearish evidence,
uncertainty notes — with SEC claims cited inline `[n]`.

Two guards (shared with P7): the **citation guard** (every `[n]` ⊆ the retrieved SEC sources)
and **number grounding** (every figure in the narrative traces to the forecast/technical/news/
SEC inputs — the model may quote, never invent). One corrective retry, then raise.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date

from pydantic import BaseModel, Field

from stock_agent.indicators.snapshot import IndicatorSnapshot
from stock_agent.llm.client import TextLLM
from stock_agent.llm.guards import NewsSummary, NumberGrounding
from stock_agent.logging_config import get_logger
from stock_agent.research._shared import correction, loads_lenient, markers_in_text
from stock_agent.research.prompts import MEMO_SYSTEM, build_memo_user
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.research import ResearchMemo, SourceCitation
from stock_agent.schemas.retrieval import EvidenceSet

log = get_logger(__name__)

_MAX_TOKENS = 2200


class MemoGuardError(RuntimeError):
    """Raised when the memo keeps citing fabricated sources / figures after a retry."""


class _RawMemo(BaseModel):
    """The raw LLM JSON shape before SEC markers are resolved to chunks."""

    executive_summary: str = ""
    management_commentary: str = ""
    business_drivers: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    bullish_evidence: list[str] = Field(default_factory=list)
    bearish_evidence: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list)


def _narrative_text(raw: _RawMemo) -> str:
    """All free-text the model wrote (for the citation + number guards to scan)."""
    return " ".join(
        [
            raw.executive_summary,
            raw.management_commentary,
            *raw.business_drivers,
            *raw.risk_factors,
            *raw.bullish_evidence,
            *raw.bearish_evidence,
            *raw.uncertainty_notes,
        ]
    )


def _quant_signal_lines(
    snapshot: IndicatorSnapshot,
    forecasts: Sequence[ScenarioForecast],
    grounding: NumberGrounding,
) -> list[str]:
    """Render forecast + technical signal lines AND seed the grounding from the same numbers."""
    lines: list[str] = []
    for fc in forecasts:
        grounding.add_from(fc.model_dump())
        var = f", VaR95={fc.var_95:.1%}" if fc.var_95 is not None else ""
        lines.append(
            f"- Forecast [{fc.model_name}, {fc.horizon_days}d]: P(up)={fc.upside_prob:.0%}, "
            f"P(down)={fc.downside_prob:.0%}, E[r]={fc.expected_return:+.1%}{var}"
        )
    grounding.add_from(snapshot.numeric_indicators())
    tech = f"- Technicals: trend={snapshot.trend_label}"
    if snapshot.rsi14 is not None:
        tech += f", RSI={snapshot.rsi14:.0f}"
    if snapshot.hist_vol_annualized is not None:
        tech += f", annualized vol={snapshot.hist_vol_annualized:.0%}"
    lines.append(tech)
    return lines


def build_memo(
    ticker: str,
    as_of: Date,
    *,
    snapshot: IndicatorSnapshot,
    forecasts: Sequence[ScenarioForecast],
    evidence: EvidenceSet,
    llm: TextLLM,
    news_summary: NewsSummary | None = None,
    max_tokens: int = _MAX_TOKENS,
) -> ResearchMemo:
    """Assemble an integrated memo: deterministic quant sections + one grounded narrative call."""
    grounding = NumberGrounding()
    quant_lines = _quant_signal_lines(snapshot, forecasts, grounding)

    news_themes: list[str] = []
    if news_summary is not None:
        grounding.add_from(news_summary.model_dump())
        news_themes = list(news_summary.key_themes[:8])  # up to 5-8 (importance-ordered)
    # Seed the grounding set from every quotable input. KNOWN LIMITATION: with ~10 full SEC
    # chunks the grounded number set is large, so this guard is recall-favoring — it reliably
    # blocks a figure absent from *all* inputs, but won't catch a fabricated number that happens
    # to coincide with an unrelated value somewhere in the sources. It is a guardrail, not a
    # precision oracle (the quant sections, which carry the load-bearing numbers, are exact).
    for rc in evidence.chunks:
        grounding.add_from(rc.chunk.text)  # SEC figures are quotable

    user = build_memo_user(ticker, as_of.isoformat(), quant_lines, news_themes, evidence.chunks)
    raw = llm.complete_json(system=MEMO_SYSTEM, user=user, max_tokens=max_tokens)
    parsed = _guarded(evidence, grounding, raw, user, llm, max_tokens, retry=True)

    citations = [
        SourceCitation(
            marker=m,
            chunk_id=evidence.chunks[m - 1].chunk.chunk_id,
            label=evidence.chunks[m - 1].citation_label(),
        )
        for m in sorted(set(parsed.citations) | markers_in_text(_narrative_text(parsed)))
    ]
    return ResearchMemo(
        ticker=ticker,
        as_of=as_of,
        technical_indicators=snapshot.numeric_indicators(),
        forecasts=list(forecasts),
        executive_summary=parsed.executive_summary,
        management_commentary=parsed.management_commentary,
        business_drivers=parsed.business_drivers,
        risk_factors=parsed.risk_factors,
        bullish_evidence=parsed.bullish_evidence,
        bearish_evidence=parsed.bearish_evidence,
        uncertainty_notes=parsed.uncertainty_notes,
        recent_news=news_themes,
        citations=citations,
    )


def _guarded(
    evidence: EvidenceSet,
    grounding: NumberGrounding,
    raw: str,
    user: str,
    llm: TextLLM,
    max_tokens: int,
    *,
    retry: bool,
) -> _RawMemo:
    """Validate the memo's citations + figures; one corrective retry, then raise."""
    parsed = _RawMemo.model_validate(loads_lenient(raw))
    n = len(evidence.chunks)
    text = _narrative_text(parsed)
    cited = set(parsed.citations) | markers_in_text(text)
    bad_cites = sorted(m for m in cited if not 1 <= m <= n)  # citations outside the retrieved set
    ungrounded = grounding.ungrounded(text)  # figures not in any input

    if bad_cites or ungrounded:
        if retry:
            log.warning("memo.guard.retry", bad_citations=bad_cites, ungrounded=ungrounded[:5])
            corrected = llm.complete_json(
                system=MEMO_SYSTEM,
                user=user + correction(bad_cites, ungrounded, n, allow_insufficient=False),
                max_tokens=max_tokens,
            )
            return _guarded(evidence, grounding, corrected, user, llm, max_tokens, retry=False)
        raise MemoGuardError(
            f"memo cited invalid sources {bad_cites} / ungrounded figures {ungrounded[:5]}"
        )
    return parsed


# ---------------------------------------------------------------------------
# Markdown render
# ---------------------------------------------------------------------------
def _bullets(items: Sequence[str]) -> list[str]:
    return [f"- {item}" for item in items]


def render_memo_markdown(memo: ResearchMemo) -> str:
    """Render a ``ResearchMemo`` as a Markdown document (numbers from modules; SEC cited)."""
    out: list[str] = [
        f"# Research Memo — {memo.ticker}",
        f"_As of {memo.as_of:%b %d, %Y}. Research/education only; not financial advice._",
        "",
        "## Executive Summary",
        memo.executive_summary or "_n/a_",
    ]

    if memo.technical_indicators:
        out += ["", "## Technical Indicators"]
        out += [f"- **{k}**: {v:.4g}" for k, v in memo.technical_indicators.items()]

    if memo.forecasts:
        out += ["", "## Probability Scenarios"]
        for fc in memo.forecasts:
            var = f", VaR95 {fc.var_95:.1%}" if fc.var_95 is not None else ""
            out.append(
                f"- **{fc.model_name}** ({fc.horizon_days}d): P(up) {fc.upside_prob:.0%}, "
                f"P(down) {fc.downside_prob:.0%}, E[r] {fc.expected_return:+.1%}{var}"
            )

    for title, items in (
        ("Recent News", memo.recent_news),
        ("Business Drivers", memo.business_drivers),
        ("Risk Factors", memo.risk_factors),
        ("Bullish Evidence", memo.bullish_evidence),
        ("Bearish Evidence", memo.bearish_evidence),
        ("Uncertainty Notes", memo.uncertainty_notes),
    ):
        if items:
            out += ["", f"## {title}", *_bullets(items)]

    if memo.management_commentary:
        out += ["", "## Management Commentary", memo.management_commentary]

    if memo.citations:
        out += ["", "## Source Citations"]
        out += [f"- [{c.marker}] {c.label}" for c in memo.citations]

    return "\n".join(out)
