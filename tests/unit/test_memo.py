"""Integrated research memo (RAG P8) — build_memo guards + assembly + render (canned LLM)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from stock_agent.indicators.snapshot import IndicatorSnapshot
from stock_agent.research.memo import MemoGuardError, build_memo, render_memo_markdown
from stock_agent.research.prompts import build_memo_user
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.forecast import ProbBucket, ScenarioForecast
from stock_agent.schemas.research import NewsAnalysis, ResearchMemo
from stock_agent.schemas.retrieval import EvidenceSet, RetrievedChunk

_AS_OF = date(2026, 2, 25)


class _FakeLLM:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        out = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return out


def _snapshot() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        as_of=_AS_OF, last_close=100.0, rsi14=55.0, hist_vol_annualized=0.45, trend_label="uptrend"
    )


def _forecast() -> ScenarioForecast:
    return ScenarioForecast(
        ticker="NVDA",
        as_of=_AS_OF,
        horizon_days=20,
        model_name="historical_sim",
        buckets=[
            ProbBucket(label="down", lower=None, upper=0.0, probability=0.4),
            ProbBucket(label="up", lower=0.0, upper=None, probability=0.6),
        ],
        expected_return=0.05,
        upside_prob=0.6,
        downside_prob=0.4,
        var_95=-0.08,
    )


def _chunk(idx: int, text: str, section: str) -> DocumentChunk:
    doc_id = "NVDA:10-K:2026-02-25"
    return DocumentChunk(
        chunk_id=f"{doc_id}:{idx}",
        document_id=doc_id,
        chunk_index=idx,
        text=text,
        ticker="NVDA",
        document_type="10-K",
        source="SEC",
        source_url="https://www.sec.gov/x",
        filing_date=_AS_OF,
        section=section,
    )


def _evidence() -> EvidenceSet:
    chunks = [
        _chunk(0, "Data Center revenue was up 41% on strong demand.", "Item 7. MD&A"),
        _chunk(1, "We depend on third-party foundries and face export-control risk.",
               "Item 1A. Risk Factors"),
    ]
    return EvidenceSet(query="memo", chunks=[RetrievedChunk(chunk=c, score=0.8) for c in chunks])


_GOOD = json.dumps(
    {
        "executive_summary": "Uptrend with 60% upside over 20 days; data center demand strong [1].",
        "management_commentary": "Management cited 41% data center revenue growth [1].",
        "business_drivers": ["Accelerated computing demand [1]"],
        "risk_factors": ["Foundry dependence and export-control exposure [2]"],
        "bullish_evidence": ["Forecast P(up) 60% aligns with demand strength [1]"],
        "bearish_evidence": ["Supply concentration risk [2]"],
        "uncertainty_notes": ["Price-only model cannot see scheduled events"],
        "citations": [1, 2],
    }
)


def _build(*responses: str, evidence: EvidenceSet | None = None) -> tuple[ResearchMemo, _FakeLLM]:
    llm = _FakeLLM(*responses)
    memo = build_memo(
        "NVDA", _AS_OF,
        snapshot=_snapshot(), forecasts=[_forecast()],
        evidence=evidence if evidence is not None else _evidence(), llm=llm,
    )
    return memo, llm


def _news() -> NewsAnalysis:
    return NewsAnalysis(
        lookback_days=21,
        article_count=12,
        overview="AI demand remains the dominant narrative.",
        key_themes=["AI capex"],
        bullish=["Hyperscaler capex guidance raised"],
        bearish=["China export restrictions widened"],
        catalysts=["GTC keynote next month"],
        risks=["Customer concentration"],
        pct_positive=0.5,
        pct_negative=0.25,
        sentiment_coverage=0.4,
    )


# ---- assembly + single call --------------------------------------------------
def test_build_memo_assembles_sections_one_call() -> None:
    memo, llm = _build(_GOOD)
    assert llm.calls == 1  # ONE synthesis call
    assert memo.ticker == "NVDA" and memo.as_of == _AS_OF
    # quant sections are copied verbatim from the modules (not from the LLM)
    assert memo.technical_indicators == _snapshot().numeric_indicators()
    assert [f.model_name for f in memo.forecasts] == ["historical_sim"]
    # narrative sections populated
    assert "data center" in memo.executive_summary.lower()
    assert memo.business_drivers and memo.risk_factors
    # citations resolved to real chunks
    assert {c.marker for c in memo.citations} == {1, 2}
    assert memo.citations[0].chunk_id == _evidence().chunks[0].chunk.chunk_id
    assert "Item 7" in memo.citations[0].label


# ---- recent-news analysis integration ----------------------------------------
def test_build_memo_stores_and_renders_news_analysis() -> None:
    llm = _FakeLLM(_GOOD)
    memo = build_memo(
        "NVDA", _AS_OF,
        snapshot=_snapshot(), forecasts=[_forecast()],
        evidence=_evidence(), llm=llm, news_analysis=_news(),
    )
    assert memo.news is not None
    assert memo.news.article_count == 12 and memo.news.lookback_days == 21
    md = render_memo_markdown(memo)
    assert "## Recent-News Analysis (12 articles, last 21d)" in md
    assert "Hyperscaler capex guidance raised" in md  # bullish insight rendered
    assert "50% positive" in md  # sentiment composition line


def test_memo_user_prompt_carries_news_insights() -> None:
    """The synthesis prompt must expose the news insights so the summary can reflect them."""
    na = _news()
    user = build_memo_user(
        "NVDA", _AS_OF.isoformat(), ["- P(up) 60%"], na.key_themes, _evidence().chunks, news=na
    )
    assert "NEWS BULLISH:" in user and "Hyperscaler capex guidance raised" in user
    assert "NEWS RISKS:" in user and "Customer concentration" in user
    # Insights sit before the SEC sources block (context, not citable evidence).
    assert user.index("NEWS BULLISH:") < user.index("SEC FILING SOURCES")


# ---- citation guard ----------------------------------------------------------
def test_fabricated_citation_retried_then_raises() -> None:
    bad = json.dumps({"executive_summary": "Per [9].", "citations": [9]})
    with pytest.raises(MemoGuardError):
        _build(bad, bad)


# ---- number grounding --------------------------------------------------------
def test_invented_number_retried_then_raises() -> None:
    # "88.8%" appears in no input (forecast/snapshot/news/SEC) -> flagged.
    bad = json.dumps({"executive_summary": "Margin reached 88.8% [1].", "citations": [1]})
    with pytest.raises(MemoGuardError):
        _build(bad, bad)


def test_grounded_numbers_pass() -> None:
    # 60% (upside_prob), 41% (SEC source) are grounded -> clean, one call.
    _, llm = _build(_GOOD)
    assert llm.calls == 1


# ---- empty evidence: still builds (no SEC citations) -------------------------
def test_empty_evidence_builds_without_citations() -> None:
    resp = json.dumps(
        {"executive_summary": "Uptrend with 60% upside.", "uncertainty_notes": ["Limited data"],
         "citations": []}
    )
    memo, llm = _build(resp, evidence=EvidenceSet(query="memo", chunks=[]))
    assert llm.calls == 1 and memo.citations == []
    assert memo.executive_summary


# ---- render ------------------------------------------------------------------
def test_render_markdown_has_sections_and_citations() -> None:
    memo, _ = _build(_GOOD)
    md = render_memo_markdown(memo)
    assert "# Research Memo — NVDA" in md
    assert "## Executive Summary" in md
    assert "## Technical Indicators" in md
    assert "## Probability Scenarios" in md
    assert "## Risk Factors" in md
    assert "## Source Citations" in md
    assert "[1] NVDA 10-K Feb 25, 2026 — Item 7. MD&A" in md
    assert "not financial advice" in md.lower()
