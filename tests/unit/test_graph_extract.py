"""Graph extraction (advanced-RAG A5.1) — triples → verified, provenance-bearing edges (canned LLM).

Offline: a scripted ``_FakeLLM`` returns triple JSON per section group; a hand-built alias map
drives resolution. Asserts: expected edges with provenance + filing metadata; 'Micron' → 'MU'
resolution; the hallucinated-edge guard drops a triple whose object is absent from the cited chunk;
confidence + self-edge filtering; the budget gate; section filtering by item-code.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from stock_agent.graph.extract import (
    GraphExtractBudgetExceeded,
    extract_edges,
    filter_graph_sections,
    resolve_company,
    select_extraction_chunks,
)
from stock_agent.schemas.documents import DocumentChunk

_ALIAS = {
    "NVDA": ["NVIDIA"], "MU": ["Micron", "Micron Technology"], "AMD": ["AMD", "Advanced Micro"],
}


class _FakeLLM:
    """Returns the next scripted JSON reply per call; records how many calls were made."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self.calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        out = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return out


def _chunk(idx: int, text: str, *, ticker: str = "NVDA", doc: str = "NVDA:10-K:2025-02-26",
           section: str = "Item 1. Business", document_type: str = "10-K") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc}:{idx}", document_id=doc, chunk_index=idx, text=text,
        ticker=ticker, document_type=document_type, source="SEC",
        source_url=f"https://www.sec.gov/Archives/edgar/data/x/{ticker.lower()}.htm",
        filing_date=date(2025, 2, 26), section=section,
    )


def _triples(*items: dict[str, object]) -> str:
    return json.dumps({"triples": list(items)})


def test_resolve_company() -> None:
    assert resolve_company("Micron", _ALIAS) == "MU"
    assert resolve_company("Micron Technology", _ALIAS) == "MU"
    assert resolve_company("MU", _ALIAS) == "MU"  # bare ticker symbol
    assert resolve_company("Some Unknown Corp", _ALIAS) is None


def test_extract_depends_on_with_provenance() -> None:
    chunks = [_chunk(0, "We purchase high-bandwidth memory from Micron and other suppliers.")]
    llm = _FakeLLM([_triples(
        {"subject": "NVIDIA", "relation": "depends_on", "object": "Micron",
         "chunk": 1, "confidence": 0.9}
    )])
    res = extract_edges("NVDA", chunks, llm=llm, alias_map=_ALIAS)
    assert len(res.edges) == 1
    e = res.edges[0]
    assert (e.subject, e.relation, e.object) == ("NVDA", "depends_on", "MU")  # resolved to ticker
    assert e.provenance == ["NVDA:10-K:2025-02-26:0"]
    assert e.filing_date == date(2025, 2, 26)
    assert e.source_url.endswith("nvda.htm")
    # both the subject (NVDA) and object (MU) nodes are registered, object carries the ticker
    ids = {ent.id: ent for ent in res.entities}
    assert ids["MU"].ticker == "MU" and ids["NVDA"].ticker == "NVDA"


def test_full_legal_name_with_punctuation_is_kept() -> None:
    # Regression: the model returns the full legal name with trailing punctuation, but the chunk
    # abbreviates it. The verifier matches the suffix-stripped CORE name (the old \b-anchored regex
    # wrongly dropped names ending in '.', e.g. "Co.", "Inc.").
    chunks = [_chunk(0, "We rely on contract manufacturers such as Hon Hai Precision Industry and "
                        "Samsung Electronics for assembly.")]
    llm = _FakeLLM([_triples(
        {"subject": "NVIDIA", "relation": "depends_on", "confidence": 0.9, "chunk": 1,
         "object": "Hon Hai Precision Industry Co., Ltd."},
        {"subject": "NVIDIA", "relation": "competes_with", "confidence": 0.8, "chunk": 1,
         "object": "Samsung Electronics Co., Ltd."},
    )])
    res = extract_edges("NVDA", chunks, llm=llm, alias_map=_ALIAS)
    objs = {(e.relation, g.name) for e in res.edges for g in res.entities if g.id == e.object}
    # both kept even though unresolved (not in the semis alias map) and named only by core form
    assert ("depends_on", "Hon Hai Precision Industry Co., Ltd.") in objs
    assert ("competes_with", "Samsung Electronics Co., Ltd.") in objs
    assert all(e.object for e in res.edges)  # normalized-name ids for unresolved companies


def test_unresolved_company_surface_variants_dedup_to_one_node() -> None:
    # F1: two surface forms of the same out-of-universe supplier (across two filings) must
    # collapse to ONE node, keyed on the suffix-stripped core name — not two nodes from raw slugs.
    c1 = _chunk(0, "We depend on supplier Hon Hai Precision Industry for assembly.",
                doc="NVDA:10-K:2024-02-21", section="Item 1. Business")
    c2 = _chunk(0, "We depend on supplier Hon Hai Precision Industry Co., Ltd. for assembly.",
                doc="NVDA:10-K:2025-02-26", section="Item 1. Business")
    llm = _FakeLLM([
        _triples({"subject": "NVIDIA", "relation": "depends_on", "chunk": 1, "confidence": 0.9,
                  "object": "Hon Hai Precision Industry"}),
        _triples({"subject": "NVIDIA", "relation": "depends_on", "chunk": 1, "confidence": 0.9,
                  "object": "Hon Hai Precision Industry Co., Ltd."}),
    ])
    res = extract_edges("NVDA", [c1, c2], llm=llm, alias_map=_ALIAS)
    obj_ids = {e.object for e in res.edges}
    assert obj_ids == {"hon-hai-precision-industry"}  # one canonical node, both filings point to it
    assert sum(1 for e in res.entities if e.type == "company" and e.ticker is None) == 1


def test_hallucinated_edge_dropped() -> None:
    # The cited chunk does NOT name the object → the verification guard drops the edge.
    chunks = [_chunk(0, "We rely on a diversified supplier base for components.")]
    llm = _FakeLLM([_triples(
        {"subject": "NVIDIA", "relation": "depends_on", "object": "Micron",
         "chunk": 1, "confidence": 0.95}
    )])
    res = extract_edges("NVDA", chunks, llm=llm, alias_map=_ALIAS)
    assert res.edges == []  # object 'Micron' absent from the provenance chunk → not admitted


def test_confidence_and_self_edge_filtering() -> None:
    chunks = [_chunk(0, "We compete with AMD; we also depend on NVIDIA internal teams.")]
    llm = _FakeLLM([_triples(
        {"subject": "NVIDIA", "relation": "competes_with", "object": "AMD",
         "chunk": 1, "confidence": 0.3},  # below default min_confidence 0.5 → dropped
        {"subject": "NVIDIA", "relation": "depends_on", "object": "NVIDIA",
         "chunk": 1, "confidence": 0.9},  # self-edge → dropped
    )])
    res = extract_edges("NVDA", chunks, llm=llm, alias_map=_ALIAS)
    assert res.edges == []


def test_risk_edge_no_substring_guard() -> None:
    # Risk objects are paraphrased; they pass on confidence + a valid chunk (no substring match).
    chunks = [_chunk(0, "A shortage of manufacturing capacity could harm our results.",
                     section="Item 1A. Risk Factors")]
    llm = _FakeLLM([_triples(
        {"subject": "NVIDIA", "relation": "mentions_risk", "object": "supply capacity shortage",
         "chunk": 1, "confidence": 0.8}
    )])
    res = extract_edges("NVDA", chunks, llm=llm, alias_map=_ALIAS)
    assert len(res.edges) == 1
    e = res.edges[0]
    assert e.relation == "mentions_risk"
    assert e.object == "supply-capacity-shortage"  # normalized node id
    nodes = {ent.id: ent for ent in res.entities}
    assert nodes["supply-capacity-shortage"].type == "risk"


def test_invalid_chunk_pointer_dropped() -> None:
    chunks = [_chunk(0, "We purchase memory from Micron.")]
    llm = _FakeLLM([_triples(
        {"subject": "NVIDIA", "relation": "depends_on", "object": "Micron",
         "chunk": 9, "confidence": 0.9}  # out of range (only 1 chunk) → dropped
    )])
    res = extract_edges("NVDA", chunks, llm=llm, alias_map=_ALIAS)
    assert res.edges == []


def test_budget_gate() -> None:
    # Two filings (two groups, each a candidate) but a 1-call ceiling → second group aborts.
    g1 = [_chunk(0, "We buy memory from Micron.", doc="NVDA:10-K:2024-02-21")]
    g2 = [_chunk(0, "We compete with AMD.", doc="NVDA:10-K:2025-02-26")]
    llm = _FakeLLM([_triples(
        {"subject": "NVIDIA", "relation": "depends_on", "object": "Micron",
         "chunk": 1, "confidence": 0.9}
    )])
    with pytest.raises(GraphExtractBudgetExceeded):
        extract_edges("NVDA", [*g1, *g2], llm=llm, alias_map=_ALIAS, max_calls=1)


def test_candidate_prefilter_skips_irrelevant() -> None:
    # No alias name, no relation cue → no LLM call is made for this group.
    chunks = [_chunk(0, "Our headquarters are located in a leased facility.")]
    llm = _FakeLLM([_triples()])
    res = extract_edges("NVDA", chunks, llm=llm, alias_map=_ALIAS)
    assert llm.calls == 0 and res.calls == 0 and res.edges == []


def test_filter_graph_sections_by_item_code() -> None:
    biz = _chunk(0, "x", section="Item 1. Business")
    risk = _chunk(1, "y", section="Item 1A. Risk Factors")
    mdna = _chunk(2, "z", section="Item 7. Management's Discussion")
    kept = filter_graph_sections(
        [biz, risk, mdna], ["Item 1. Business", "Item 1A. Risk Factors"]
    )
    assert {c.section for c in kept} == {"Item 1. Business", "Item 1A. Risk Factors"}


def test_select_chunks_prefers_sections_when_detected() -> None:
    # Clean section detection (>= _MIN_SECTION_CHUNKS Item 1/1A chunks) → use exactly those.
    biz = [_chunk(i, "We compete with AMD.", section="Item 1. Business") for i in range(6)]
    other = [_chunk(99, "cash flow table", section="Item 8. Financial Statements")]
    sel = select_extraction_chunks(biz + other, ["Item 1. Business"], _ALIAS)
    assert len(sel) == 6 and all(c.section == "Item 1. Business" for c in sel)


def test_select_chunks_falls_back_to_whole_10k_when_sections_collapse() -> None:
    # INTC-style: section detection failed (everything 'Preamble', only a stray TOC stub matches).
    # Fall back to cue-bearing chunks; a pure-number chunk with no company/risk cue is excluded.
    stub = _chunk(0, "Item 1A. Risk Factors Pages 37-51", section="Item 1A.Risk FactorsPages 37-51")
    body_cue = [
        _chunk(i, "We face competition from AMD and supply risk.", section="Preamble")
        for i in range(1, 4)
    ]
    numbers_only = _chunk(50, "1,234 5,678 9,012", section="Preamble")
    sel = select_extraction_chunks(
        [stub, *body_cue, numbers_only], ["Item 1. Business", "Item 1A. Risk Factors"], _ALIAS
    )
    sel_ids = {c.chunk_id for c in sel}
    assert all(c.chunk_id in sel_ids for c in body_cue)  # cue-bearing Preamble chunks included
    assert numbers_only.chunk_id not in sel_ids  # no cue → excluded


def test_20f_routed_through_fallback_not_section_matched() -> None:
    # A foreign 20-F has business + risk factors but a DIFFERENT item structure: its "Item 1" is
    # *Identity of Directors*, which shares code "1" with a 10-K's "Item 1. Business". 20-F must NOT
    # be section-matched (would mis-tag Directors as Business); it always uses the cue-bearing
    # fallback. Need pool >= _MIN_SECTION_CHUNKS to trigger the fallback, so build 5 cue chunks + 1.
    cue = [
        _chunk(i, "We rely on suppliers including ASML for lithography equipment and face risk.",
               document_type="20-F", section="Item 4. Information on the Company",
               doc="TSM:20-F:2025-03-15")
        for i in range(5)
    ]
    directors = _chunk(99, "Identity of directors and senior management.", document_type="20-F",
                       section="Item 1. Identity of Directors", doc="TSM:20-F:2025-03-15")
    sel = select_extraction_chunks(
        [*cue, directors], ["Item 1. Business", "Item 1A. Risk Factors"], _ALIAS
    )
    ids = {c.chunk_id for c in sel}
    assert all(c.chunk_id in ids for c in cue)  # cue-bearing 20-F business prose kept by fallback
    assert directors.chunk_id not in ids  # no cue, and NOT matched as 10-K "Item 1"


def test_filter_excludes_10q_8k_sharing_item_code_one() -> None:
    # Regression: 10-Q "Item 1" (Financial Statements) and 8-K "Item 1.01" share the code "1" with
    # 10-K "Item 1. Business" — they must NOT pass (doc-type gate + decimal-subitem lookahead).
    tenk_biz = _chunk(0, "x", section="Item 1. Business", document_type="10-K")
    tenq_fin = _chunk(1, "y", section="Item 1. Financial Statements", document_type="10-Q",
                      doc="NVDA:10-Q:2025-05-28")
    eightk = _chunk(2, "z", section="Item 1.01 Entry into a Material Agreement",
                    document_type="8-K", doc="NVDA:8-K:2025-03-01")
    kept = filter_graph_sections(
        [tenk_biz, tenq_fin, eightk], ["Item 1. Business", "Item 1A. Risk Factors"]
    )
    assert [c.document_type for c in kept] == ["10-K"]  # only the 10-K business section survives
