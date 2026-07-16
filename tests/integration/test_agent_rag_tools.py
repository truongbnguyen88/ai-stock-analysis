"""RAG agent tools (P8.5) — search_filings + research_summary (offline, no network).

Both tools expose the GUARDED synthesis (P7 ``answer_question`` / P8 ``run_research``), not
raw retrieval, so the citation guard + number grounding stay intact. These tests inject a fake
``Retriever`` (FakeEmbedder + InMemoryVectorStore) and a canned ``TextLLM`` for ``search_filings``,
and monkeypatch ``run_research`` for ``research_summary`` — no embedding model is ever loaded and
no LLM/network call is made.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from stock_agent.agent.runtime import ToolResponse, ToolUse, run_agent
from stock_agent.agent.tools import TOOL_SCHEMAS, ToolExecutor
from stock_agent.pipelines.research import ResearchPipelineError
from stock_agent.rag.embeddings import FakeEmbedder
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.vector_store import InMemoryVectorStore
from stock_agent.schemas.documents import DocumentChunk, DocumentType
from stock_agent.schemas.earnings import EarningsContext
from stock_agent.schemas.forecast import LargeMoveBreakdown, ScenarioForecast
from stock_agent.schemas.research import (
    NewsAnalysis,
    PriceSnapshot,
    ResearchMemo,
    SourceCitation,
)
from stock_agent.settings import Settings

_EMB = FakeEmbedder(dim=32)


# ---- fakes -------------------------------------------------------------------
class _FakeLLM:
    """``TextLLM`` returning canned JSON in order (last repeats); records call count."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        out = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return out


class _Scripted:
    """``ToolLLM`` that replays a fixed list of ``ToolResponse`` turns."""

    def __init__(self, responses: list[ToolResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, *, system, messages, tools, max_tokens) -> ToolResponse:  # type: ignore[no-untyped-def]
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


def _answer_json(answer: str, citations: list[int]) -> str:
    return json.dumps({"answer": answer, "citations": citations, "insufficient_evidence": False})


def _decision_json(action: str, query: str | None = None, ticker: str | None = None) -> str:
    """A canned A4 ReAct decision (the loop's per-step structured output)."""
    d: dict[str, object] = {"thought": "t", "action": action}
    if query is not None:
        d["query"] = query
    if ticker is not None:
        d["ticker"] = ticker
    return json.dumps(d)


def _chunk(
    idx: int, text: str, *, ticker: str = "NVDA", dtype: DocumentType = "10-K"
) -> DocumentChunk:
    doc_id = f"{ticker}:{dtype}:2026-02-25"
    return DocumentChunk(
        chunk_id=f"{doc_id}:{idx}",
        document_id=doc_id,
        chunk_index=idx,
        text=text,
        ticker=ticker,
        document_type=dtype,
        source="SEC",
        source_url="https://www.sec.gov/x",
        filing_date=date(2026, 2, 25),
        section="Item 1A. Risk Factors",
    )


def _retriever(chunks: list[DocumentChunk]) -> Retriever:
    store = InMemoryVectorStore()
    if chunks:
        store.add(chunks, _EMB.embed_documents([c.text for c in chunks]))
    return Retriever(_EMB, store)


def _executor(*, llm: _FakeLLM | None, retriever: Retriever | None = None) -> ToolExecutor:
    # Inject via the explicit `retriever=` override so no real embedder/graph is ever built.
    return ToolExecutor(Settings(_env_file=None), llm=llm, retriever=retriever)


def _final(text: str) -> ToolResponse:
    return ToolResponse(text=text, tool_uses=[], stop_reason="end_turn", assistant_content=[])


# ---- schema conformance -------------------------------------------------------
def test_both_tool_schemas_present() -> None:
    by_name = {t["name"]: t for t in TOOL_SCHEMAS}
    assert {"search_filings", "research_summary", "research_multistep"} <= set(by_name)
    sf = by_name["search_filings"]["input_schema"]
    assert sf["required"] == ["ticker", "question"]
    assert {"ticker", "question", "top_k"} <= set(sf["properties"])
    rs = by_name["research_summary"]["input_schema"]
    assert rs["required"] == ["ticker"]
    assert {"ticker", "days"} <= set(rs["properties"])
    rm = by_name["research_multistep"]["input_schema"]
    assert rm["required"] == ["question"]
    assert "question" in rm["properties"]


# ---- search_filings -----------------------------------------------------------
def test_search_filings_returns_cited_answer() -> None:
    # The grounded figure (41%) is present in the source chunk, so the P7 guards pass.
    chunk_text = "Data Center revenue was up 41% from a year ago on strong demand."
    llm = _FakeLLM(_answer_json("Data Center revenue rose 41% [1].", [1]))
    ex = _executor(llm=llm, retriever=_retriever([_chunk(0, chunk_text)]))

    out = ex.execute("search_filings", {"ticker": "NVDA", "question": "How did Data Center do?"})

    assert "error" not in out
    assert out["insufficient_evidence"] is False
    assert out["n_sources"] == 1
    assert "41%" in out["answer"]
    assert len(out["citations"]) == 1
    cit = out["citations"][0]
    assert cit["marker"] == 1
    assert cit["chunk_id"] == "NVDA:10-K:2026-02-25:0"
    assert "NVDA 10-K" in cit["label"]


def test_search_filings_empty_store_is_insufficient_with_hint() -> None:
    llm = _FakeLLM(_answer_json("should not be used", [1]))
    ex = _executor(llm=llm, retriever=_retriever([]))  # nothing ingested

    out = ex.execute("search_filings", {"ticker": "TSLA", "question": "risk factors?"})

    assert out["insufficient_evidence"] is True
    assert out["answer"] == "Insufficient evidence found."
    assert out["n_sources"] == 0
    assert "ingest" in out["hint"].lower() and "TSLA" in out["hint"]
    assert llm.calls == 0  # empty retrieval short-circuits — no paid LLM call


def test_search_filings_without_llm_errors() -> None:
    ex = _executor(llm=None, retriever=_retriever([_chunk(0, "anything")]))
    out = ex.execute("search_filings", {"ticker": "NVDA", "question": "q"})
    assert "error" in out and "LLM" in out["error"]


# ---- research_multistep (A4) --------------------------------------------------
def _multihop_llm() -> _FakeLLM:
    # The canned LLM serves the loop's decisions in order, then the terminal synthesis: the call
    # sequence is search(NVDA) → search(AMD) → stop → terminal answer (one answer_question call).
    # The explicit "stop" makes this robust to agentic_max_steps (>=2): the loop halts after the
    # two hops, and the terminal synthesis gets the answer JSON (not a misparsed decision).
    return _FakeLLM(
        _decision_json("search", "export control risk", "NVDA"),
        _decision_json("search", "export control risk", "AMD"),
        _decision_json("stop"),
        _answer_json("Both NVDA [1] and AMD [2] flag export-control risk.", [1, 2]),
    )


def test_research_multistep_returns_cited_multihop_answer() -> None:
    nvda = _chunk(0, "NVDA faces export-control risk on advanced GPUs.", ticker="NVDA")
    amd = _chunk(1, "AMD faces export-control risk on AI accelerators.", ticker="AMD")
    ex = _executor(llm=_multihop_llm(), retriever=_retriever([nvda, amd]))

    out = ex.execute(
        "research_multistep", {"question": "Compare NVDA and AMD export-control risk."}
    )

    assert "error" not in out
    assert out["insufficient_evidence"] is False
    assert out["n_steps"] == 2  # two search hops (NVDA, AMD) before stop
    assert out["n_evidence"] == 2  # the per-ticker filters yield one chunk each, union of 2
    assert {c["marker"] for c in out["citations"]} == {1, 2}
    # Per-step trace is surfaced, scoped to the right ticker each hop.
    assert [t["ticker"] for t in out["trace"]] == ["NVDA", "AMD"]


def test_research_multistep_empty_corpus_is_insufficient_with_hint() -> None:
    # Nothing ingested → every step retrieves nothing → empty union → P7 refusal with NO LLM
    # synthesis call (only the loop's decision calls happen).
    llm = _FakeLLM(_decision_json("search", "risk", "TSLA"), _decision_json("stop"))
    ex = _executor(llm=llm, retriever=_retriever([]))

    out = ex.execute("research_multistep", {"question": "Compare TSLA 2023 vs 2025 risks."})

    assert out["insufficient_evidence"] is True
    assert out["n_evidence"] == 0
    assert "ingest" in out["hint"].lower()


def test_research_multistep_blank_question_errors() -> None:
    ex = _executor(llm=_FakeLLM(), retriever=_retriever([]))
    out = ex.execute("research_multistep", {"question": "   "})
    assert "error" in out and "non-empty" in out["error"]


def test_research_multistep_without_llm_errors() -> None:
    ex = _executor(llm=None, retriever=_retriever([]))
    out = ex.execute("research_multistep", {"question": "compare X and Y"})
    assert "error" in out and "LLM" in out["error"]


# ---- A5.3 multi-hop routing (graph vs hybrid) --------------------------------
def test_multistep_routes_to_graph_when_enabled_and_no_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Default policy: the multi-hop path builds + memoizes the GraphRetriever (single-shot stays
    # hybrid). We stub build_graph_system so no real graph DB / embedder is touched.
    graph = _retriever([])
    calls = {"n": 0}

    def _fake_build(settings: object) -> Retriever:
        calls["n"] += 1
        return graph

    monkeypatch.setattr("stock_agent.agent.tools.build_graph_system", _fake_build)
    ex = ToolExecutor(Settings(_env_file=None), llm=None)  # graph_multistep_enabled defaults True
    assert ex._get_multistep_retriever() is graph
    assert ex._get_multistep_retriever() is graph  # memoized
    assert calls["n"] == 1  # built once


def test_multistep_prefers_injected_base_over_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    # An injected base retriever (tests / explicit) short-circuits graph construction.
    def _boom(settings: object) -> Retriever:
        raise AssertionError("build_graph_system must not be called when a base is injected")

    monkeypatch.setattr("stock_agent.agent.tools.build_graph_system", _boom)
    base = _retriever([])
    ex = _executor(llm=None, retriever=base)
    assert ex._get_multistep_retriever() is base


def test_multistep_disabled_uses_hybrid(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(settings: object) -> Retriever:
        raise AssertionError("graph disabled: build_graph_system must not be called")

    monkeypatch.setattr("stock_agent.agent.tools.build_graph_system", _boom)
    base = _retriever([])
    ex = ToolExecutor(Settings(_env_file=None, graph_multistep_enabled=False), llm=None)
    ex._retriever = base  # disabled path returns the hybrid retriever
    assert ex._get_multistep_retriever() is base


def test_multistep_falls_back_to_hybrid_when_graph_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Graph wiring failure (e.g. no graph DB) must degrade to hybrid, never break QA.
    def _raise(settings: object) -> Retriever:
        raise RuntimeError("no graph store")

    hybrid = _retriever([])
    monkeypatch.setattr("stock_agent.agent.tools.build_graph_system", _raise)
    ex = ToolExecutor(Settings(_env_file=None), llm=None)  # _retriever is None
    monkeypatch.setattr(ex, "_get_retriever", lambda: hybrid)
    assert ex._get_multistep_retriever() is hybrid


def test_multistep_routes_to_graph_even_after_single_shot_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: a prior search_filings / research_summary fills the lazy single-shot cache
    # (`_retriever`). The multi-hop path must STILL route to graph — the filled cache must not
    # divert it to hybrid. (Before the fix, `_get_multistep_retriever` short-circuited on
    # `_retriever`, so graph activation was silently order-dependent within a session.)
    graph = _retriever([])
    monkeypatch.setattr("stock_agent.agent.tools.build_graph_system", lambda s: graph)
    ex = ToolExecutor(Settings(_env_file=None), llm=None)  # no injection; graph enabled by default
    ex._retriever = _retriever([])  # simulate a single-shot tool having run earlier this session
    assert ex._get_multistep_retriever() is graph  # graph, NOT the single-shot cache


# ---- research_summary ---------------------------------------------------------
def _memo() -> ResearchMemo:
    return ResearchMemo(
        ticker="NVDA",
        as_of=date(2026, 2, 25),
        technical_indicators={"rsi_14": 55.0, "hist_vol_20": 0.42},
        forecasts=[
            ScenarioForecast(
                ticker="NVDA",
                horizon_days=20,
                model_name="historical_sim",
                as_of=date(2026, 2, 25),
                expected_return=0.03,
                upside_prob=0.76,
                downside_prob=0.24,
                var_95=-0.08,
                buckets=[],
            )
        ],
        executive_summary="NVDA shows strong Data Center momentum.",
        business_drivers=["Data Center demand [1]"],
        risk_factors=["Export controls [1]"],
        bullish_evidence=["Record revenue [1]"],
        bearish_evidence=["Customer concentration [1]"],
        uncertainty_notes=["Pace of AI capex"],
        recent_news=["AI demand"],
        price_snapshot=PriceSnapshot(
            window_days=30, n_bars=20, first_close=90.0, last_close=100.0,
            period_high=110.0, period_low=85.0, pct_change=0.111, last_return=-0.02,
        ),
        large_move=LargeMoveBreakdown(
            ticker="NVDA", as_of=date(2026, 2, 25), horizon_days=20, model_name="ensemble",
            threshold=0.05, prob_large_move=0.9, prob_big_up=0.5, prob_big_down=0.4, lean="up",
        ),
        earnings=EarningsContext(
            ticker="NVDA", as_of=date(2026, 2, 25), next_earnings_date=date(2026, 5, 20),
            last_earnings_date=date(2026, 2, 19), days_to_next_earnings=84,
            horizon_days=20, earnings_in_horizon=False,
        ),
        news=NewsAnalysis(
            lookback_days=21,
            article_count=8,
            overview="AI demand dominates.",
            key_themes=["AI demand"],
            bullish=["Capex guidance raised"],
            risks=["Export controls"],
            pct_positive=0.5,
            pct_negative=0.25,
            sentiment_coverage=0.4,
            avg_sentiment=0.16,
        ),
        citations=[
            SourceCitation(
                marker=1,
                chunk_id="NVDA:10-K:2026-02-25:0",
                label="NVDA 10-K Feb 25, 2026 — Item 1A. Risk Factors",
            )
        ],
    )


def test_research_summary_compact_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stock_agent.agent.tools.run_research", lambda *a, **k: _memo())
    ex = _executor(llm=_FakeLLM(), retriever=_retriever([]))

    out = ex.execute("research_summary", {"ticker": "NVDA"})

    assert "error" not in out
    assert out["ticker"] == "NVDA"
    assert out["executive_summary"]
    for key in ("business_drivers", "risk_factors", "bullish_evidence", "bearish_evidence",
                "uncertainty_notes", "recent_news"):
        assert isinstance(out[key], list)
    # Headline forecasts carry the model's numbers (the LLM never produces these).
    fc = out["forecasts"][0]
    assert {"model", "horizon_days", "prob_up", "expected_return", "var_95"} <= set(fc)
    assert fc["prob_up"] == 0.76
    assert out["technical_indicators"]["rsi_14"] == 55.0
    assert out["citations"][0]["chunk_id"] == "NVDA:10-K:2026-02-25:0"
    # Recent-news analysis block surfaced (themes + insights + sentiment) for chart + agent use.
    assert out["news"]["lookback_days"] == 21 and out["news"]["article_count"] == 8
    assert out["news"]["bullish"] == ["Capex guidance raised"]
    assert out["news"]["pct_positive"] == 0.5
    assert out["news"]["avg_sentiment"] == 0.16  # drives the Net-sentiment card
    # Consolidated brief signals (were separate tools) → feed the 5 cards + large-move chart.
    assert out["price_snapshot"]["last_close"] == 100.0
    assert out["large_move"]["prob_large_move"] == 0.9 and out["large_move"]["lean"] == "up"
    assert out["earnings"]["next_earnings_date"] == "2026-05-20"  # JSON-mode date string
    assert out["earnings"]["earnings_in_horizon"] is False
    # The full Markdown is NOT returned (too long for a tool result).
    assert "##" not in json.dumps(out)


def test_research_summary_defaults_to_21_day_news_window(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _capture(*a: object, **k: object) -> ResearchMemo:
        captured.update(k)
        return _memo()

    monkeypatch.setattr("stock_agent.agent.tools.run_research", _capture)
    ex = _executor(llm=_FakeLLM(), retriever=_retriever([]))
    ex.execute("research_summary", {"ticker": "NVDA"})
    assert captured["days"] == 21  # brief pulls 3 weeks of news by default


def test_research_summary_without_llm_errors() -> None:
    ex = _executor(llm=None, retriever=_retriever([]))
    out = ex.execute("research_summary", {"ticker": "NVDA"})
    assert "error" in out and "LLM" in out["error"]


def test_research_summary_pipeline_error_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise ResearchPipelineError("no SEC evidence ingested")

    monkeypatch.setattr("stock_agent.agent.tools.run_research", _boom)
    ex = _executor(llm=_FakeLLM(), retriever=_retriever([]))
    out = ex.execute("research_summary", {"ticker": "NVDA"})
    assert "error" in out and "ingested" in out["error"]


# ---- agent-loop routing + grounding ------------------------------------------
def test_agent_routes_filing_question_to_search_filings() -> None:
    chunk_text = "Data Center revenue was up 41% from a year ago on strong demand."
    llm = _FakeLLM(_answer_json("Revenue rose 41% [1].", [1]))
    ex = _executor(llm=llm, retriever=_retriever([_chunk(0, chunk_text)]))

    script = [
        ToolResponse(
            text="",
            tool_uses=[
                ToolUse(id="1", name="search_filings",
                        input={"ticker": "NVDA", "question": "Data Center growth?"})
            ],
            stop_reason="tool_use",
            assistant_content=[],
        ),
        # Final answer relays the tool's grounded figure — the agent grounding guard accepts it.
        _final("Per the filings, Data Center revenue rose 41%."),
    ]
    result = run_agent(
        "What do NVDA's filings say about Data Center?", llm=_Scripted(script), executor=ex
    )
    assert "search_filings" in result.tool_calls
    assert "41%" in result.text  # grounded from the tool output; no grounding error raised


def test_agent_routes_overview_to_research_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("stock_agent.agent.tools.run_research", lambda *a, **k: _memo())
    ex = _executor(llm=_FakeLLM(), retriever=_retriever([]))

    script = [
        ToolResponse(
            text="",
            tool_uses=[ToolUse(id="1", name="research_summary", input={"ticker": "NVDA"})],
            stop_reason="tool_use",
            assistant_content=[],
        ),
        # Relays a model number from the memo (P(up) 76%) — accepted by the grounding guard.
        _final("The 20-day model puts P(up) at 76%. NVDA shows strong Data Center momentum."),
    ]
    result = run_agent("Give me the full picture on NVDA.", llm=_Scripted(script), executor=ex)
    assert "research_summary" in result.tool_calls
    assert "76%" in result.text


def test_agent_routes_comparative_to_research_multistep() -> None:
    nvda = _chunk(0, "NVDA faces export-control risk on advanced GPUs.", ticker="NVDA")
    amd = _chunk(1, "AMD faces export-control risk on AI accelerators.", ticker="AMD")
    ex = _executor(llm=_multihop_llm(), retriever=_retriever([nvda, amd]))

    script = [
        ToolResponse(
            text="",
            tool_uses=[
                ToolUse(
                    id="1",
                    name="research_multistep",
                    input={"question": "Compare NVDA and AMD export-control risk."},
                )
            ],
            stop_reason="tool_use",
            assistant_content=[],
        ),
        _final("Per the filings, both NVDA and AMD flag export-control risk."),
    ]
    result = run_agent(
        "Compare NVDA's and AMD's export-control risk factors.",
        llm=_Scripted(script),
        executor=ex,
    )
    assert "research_multistep" in result.tool_calls


def test_get_retriever_builds_logs_and_memoizes(monkeypatch: pytest.MonkeyPatch) -> None:
    # The observability log added in _get_retriever (active embedder + collection + chunk count)
    # must not break the build, and the retriever stays session-memoized (built once).
    import stock_agent.agent.tools as tools_mod

    store = InMemoryVectorStore()
    # _get_retriever builds the store directly (for the count log) and delegates retriever
    # construction to build_retrieval_system, which builds the embedder via rag.read_path.
    monkeypatch.setattr(tools_mod, "build_vector_store", lambda s: store)
    monkeypatch.setattr("stock_agent.rag.read_path.build_embedder", lambda s: _EMB)
    # Pin to dense so this memoization/logging test stays hermetic (no real FTS5 sparse DB on disk);
    # the hybrid default is covered by the read-path unit tests.
    ex = ToolExecutor(Settings(_env_file=None, retrieval_mode="dense"), llm=None)

    retriever = ex._get_retriever()
    # dense + rerank off → build_retrieval_system returns the plain dense Retriever.
    assert isinstance(retriever, Retriever)
    assert ex._get_retriever() is retriever  # memoized — not rebuilt
