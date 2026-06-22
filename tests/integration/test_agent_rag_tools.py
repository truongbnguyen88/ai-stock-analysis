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
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.research import ResearchMemo, SourceCitation
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
    ex = ToolExecutor(Settings(_env_file=None), llm=llm)
    if retriever is not None:
        ex._retriever = retriever  # inject so _get_retriever never builds a real embedder
    return ex


def _final(text: str) -> ToolResponse:
    return ToolResponse(text=text, tool_uses=[], stop_reason="end_turn", assistant_content=[])


# ---- schema conformance -------------------------------------------------------
def test_both_tool_schemas_present() -> None:
    by_name = {t["name"]: t for t in TOOL_SCHEMAS}
    assert {"search_filings", "research_summary"} <= set(by_name)
    sf = by_name["search_filings"]["input_schema"]
    assert sf["required"] == ["ticker", "question"]
    assert {"ticker", "question", "top_k"} <= set(sf["properties"])
    rs = by_name["research_summary"]["input_schema"]
    assert rs["required"] == ["ticker"]
    assert {"ticker", "days"} <= set(rs["properties"])


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
    # The full Markdown is NOT returned (too long for a tool result).
    assert "##" not in json.dumps(out)


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
