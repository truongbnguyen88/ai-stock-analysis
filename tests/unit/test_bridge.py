"""Deterministic entity-bridge (A4 bridging fix) — offline (fake retriever + injected alias map).

The loop won't pivot to a discovered related entity; this pass does it structurally. Tests cover the
gate (bridging cue), alias resolution, relevance ranking + cap, corpus-absence skip, and the
end-to-end `answer_multistep` integration (a bridging question now reaches the supplier's filings).
"""

from __future__ import annotations

import json
from datetime import date

from stock_agent.research.agentic import answer_multistep
from stock_agent.research.bridge import (
    bridge_candidates,
    is_bridging,
    mentioned_tickers,
)
from stock_agent.research.prompts import REACT_SYSTEM
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

_ALIASES = {"MU": ["micron"], "AMD": ["advanced micro devices"], "MSFT": ["microsoft"]}


def _chunk(cid: str, text: str, ticker: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=cid, document_id=cid.rsplit(":", 1)[0], chunk_index=0, text=text,
            ticker=ticker, document_type="10-K", source="SEC", source_url="https://sec.gov/x",
            filing_date=date(2026, 2, 25), section="Item 1A. Risk Factors",
        ),
        score=score,
    )


class _ByTickerRetriever:
    """Returns canned chunks per scoped ticker; records the tickers it was asked to search."""

    name = "fake"

    def __init__(self, by_ticker: dict[str, list[RetrievedChunk]]) -> None:
        self._by_ticker = by_ticker
        self.scoped: list[str] = []

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        tic = where.ticker if where else None
        if tic:
            self.scoped.append(tic)
        return EvidenceSet(query=query, chunks=self._by_ticker.get(tic or "", []))


# ---- gate + resolution -------------------------------------------------------
def test_is_bridging_cues() -> None:
    assert is_bridging("which of NVDA's suppliers flags the same risk in its own filings?")
    assert is_bridging("do any of its customers disclose this?")
    # single-entity question with no relationship/own-filing cue -> not a bridge
    assert not is_bridging("what does NVDA say about data center demand and competition?")


def test_mentioned_tickers_resolves_names() -> None:
    text = "We depend on Micron and Microsoft; we compete with Advanced Micro Devices."
    assert mentioned_tickers(text, _ALIASES) == {"MU", "AMD", "MSFT"}
    assert mentioned_tickers("no companies here", _ALIASES) == set()


# ---- candidate ranking + cap -------------------------------------------------
def _bridge_corpus() -> _ByTickerRetriever:
    # Each entity's own filings, with MU the most question-relevant (highest score).
    return _ByTickerRetriever(
        {
            "MU": [_chunk("MU:10-K:1", "Micron faces a CAC action and export curbs.", "MU", 0.9)],
            "AMD": [_chunk("AMD:10-K:1", "AMD competes in accelerators.", "AMD", 0.5)],
            "MSFT": [_chunk("MSFT:10-K:1", "Microsoft cloud demand.", "MSFT", 0.4)],
        }
    )


def test_bridge_candidates_ranks_and_caps() -> None:
    # Union names Micron, AMD, Microsoft; cap=2 keeps the two most relevant, MU first.
    union = [_chunk("NVDA:10-K:1", "We rely on Micron; we compete with Advanced Micro Devices "
                    "and partner with Microsoft.", "NVDA", 0.8)]
    out = bridge_candidates(
        "which supplier discloses government restrictions in its own filings?", union,
        retriever=_bridge_corpus(), alias_map=_ALIASES, searched={"NVDA"},
        per_step_k=6, max_entities=2,
    )
    assert [t for t, _ in out] == ["MU", "AMD"]  # ranked by score, capped at 2


def test_bridge_no_op_when_not_bridging() -> None:
    union = [_chunk("NVDA:10-K:1", "We rely on Micron.", "NVDA", 0.8)]
    out = bridge_candidates(
        "what is NVDA's data center demand?", union,  # no bridge cue
        retriever=_bridge_corpus(), alias_map=_ALIASES, searched={"NVDA"},
        per_step_k=6, max_entities=2,
    )
    assert out == []


def test_bridge_skips_already_searched_and_absent() -> None:
    union = [_chunk("NVDA:10-K:1", "We rely on Micron and Microsoft.", "NVDA", 0.8)]
    # MSFT has no chunks (absent from the corpus); AMD is excluded as already searched.
    retr = _ByTickerRetriever({"MU": [_chunk("MU:10-K:1", "Micron risk.", "MU", 0.9)]})
    out = bridge_candidates(
        "which supplier flags risk in its own filings?", union,
        retriever=retr, alias_map=_ALIASES, searched={"NVDA", "AMD"},  # AMD excluded as searched
        per_step_k=6, max_entities=3,
    )
    assert [t for t, _ in out] == ["MU"]  # MSFT skipped (no chunks), AMD skipped (searched)


# ---- end-to-end: answer_multistep bridges to the supplier --------------------
class _LoopLLM:
    """Stateless: drives the loop to gather NVDA's supplier list then stop; serves the answer."""

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        if system != REACT_SYSTEM:
            return json.dumps({"answer": "Micron flags a CAC action [2].", "citations": [2]})
        if "EVIDENCE GATHERED (0 chunks)" in user:
            return json.dumps({"thought": "find suppliers", "action": "search",
                               "query": "memory suppliers", "ticker": "NVDA"})
        return json.dumps({"thought": "have suppliers", "action": "stop"})


def test_answer_multistep_bridges_to_supplier_filings() -> None:
    nvda = _chunk("NVDA:10-K:1", "We depend on Micron for memory.", "NVDA", 0.8)
    mu = _chunk("MU:10-K:1", "Micron faces a CAC action and China export restrictions.", "MU", 0.9)
    retriever = _ByTickerRetriever({"NVDA": [nvda], "MU": [mu]})
    res = answer_multistep(
        "Which of NVDA's named suppliers discloses government restrictions in its own filings?",
        settings=Settings(_env_file=None), llm=_LoopLLM(), retriever=retriever,
        alias_map=_ALIASES,
    )
    union_tickers = {rc.chunk.ticker for rc in res.evidence}
    assert "MU" in union_tickers  # the bridge reached Micron's OWN filings (the loop never would)
    assert "MU" in retriever.scoped  # a scoped retrieval to Micron actually happened
    assert res.n_steps == 1  # n_steps counts the LLM's loop hops, NOT the auto entity-bridge
    assert any("entity-bridge" in st.thought for st in res.trace)  # bridge recorded in the trace
