"""Graph retrieval (advanced-RAG A5.2) — traversal ⊕ vector, fused by RRF (fakes only, offline).

A fake ``GraphStore`` (the NVDA→MU edge) + a fake base ``RetrievalSystem`` (per-ticker canned
hits) + a dict ``chunk_fetch``. Asserts: the bridging hop surfaces the neighbor's chunk a base
query misses; provenance chunks are materialized and present; no seed / no edges degrade to the base
result; the union dedups; ``RetrievalSystem`` conformance.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from stock_agent.graph.retriever import GraphRetriever
from stock_agent.graph.store import Direction, GraphCounts
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.graph import Edge, Entity, Relation
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk

_ALIAS = {"NVDA": ["NVIDIA"], "MU": ["Micron"]}


def _chunk(cid: str, text: str, ticker: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid, document_id=f"{ticker}:10-K:2025-02-26", chunk_index=0, text=text,
        ticker=ticker, document_type="10-K", source="SEC",
        source_url=f"https://sec.gov/{ticker}.htm", filing_date=date(2025, 2, 26),
        section="Item 1A. Risk Factors",
    )


_NVDA_BASE = _chunk("nvda:1", "NVDA discloses supply risk.", "NVDA")
_NVDA_PROV = _chunk("nvda:prov", "We purchase memory from Micron.", "NVDA")
_MU_RISK = _chunk("mu:risk", "Micron faces memory capacity oversupply risk.", "MU")


class _FakeBase:
    """Per-ticker canned retrieval; an unfiltered query returns only the NVDA base chunk."""

    name = "fakebase"

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        if where is not None and where.ticker == "MU":
            return EvidenceSet(query=query, chunks=[RetrievedChunk(chunk=_MU_RISK, score=0.7)])
        if where is not None and where.ticker == "NVDA":
            return EvidenceSet(query=query, chunks=[RetrievedChunk(chunk=_NVDA_BASE, score=0.8)])
        return EvidenceSet(query=query, chunks=[RetrievedChunk(chunk=_NVDA_BASE, score=0.8)])


class _FakeGraph:
    """One-edge graph implementing the GraphStore Protocol (only neighbors/get_entity are used)."""

    def __init__(self, edges: list[Edge], entities: dict[str, Entity]) -> None:
        self._edges = edges
        self._entities = entities

    def neighbors(
        self, entity_id: str, *, relations: Sequence[Relation] | None = None,
        hops: int = 1, direction: Direction = "out",
    ) -> list[Edge]:
        return [e for e in self._edges if e.subject == entity_id]

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    # Unused by GraphRetriever — present so _FakeGraph structurally satisfies GraphStore.
    def add_entities(self, entities: Sequence[Entity]) -> None:
        raise NotImplementedError

    def add_edges(self, edges: Sequence[Edge]) -> None:
        raise NotImplementedError

    def provenance_chunk_ids(
        self, entity_id: str, *, relations: Sequence[Relation] | None = None,
        hops: int = 1, direction: Direction = "out",
    ) -> list[str]:
        raise NotImplementedError

    def count(self) -> GraphCounts:
        return GraphCounts(nodes=0, edges=0)


def _edge() -> Edge:
    return Edge(
        subject="NVDA", relation="depends_on", object="MU", provenance=["nvda:prov"],
        filing_date=date(2025, 2, 26), source_url="https://sec.gov/NVDA.htm", confidence=0.9,
    )


def _fetch(ids: Sequence[str]) -> dict[str, DocumentChunk]:
    table = {"nvda:prov": _NVDA_PROV, "nvda:1": _NVDA_BASE, "mu:risk": _MU_RISK}
    return {i: table[i] for i in ids if i in table}


def _retriever() -> GraphRetriever:
    graph = _FakeGraph(
        [_edge()],
        {"MU": Entity(id="MU", name="Micron", type="company", ticker="MU"),
         "NVDA": Entity(id="NVDA", name="NVIDIA", type="company", ticker="NVDA")},
    )
    return GraphRetriever(graph, _FakeBase(), _fetch, alias_map=_ALIAS)


def test_conformance() -> None:
    assert isinstance(_retriever(), RetrievalSystem)


def test_bridging_surfaces_neighbor_and_provenance() -> None:
    # A query about NVDA: base alone returns only NVDA's chunk; the graph adds Micron's risk chunk
    # (scoped "what") and the relationship-stating provenance chunk (the "who").
    ev = _retriever().retrieve("supply risk", top_k=5, where=ChunkFilter(ticker="NVDA"))
    ids = {rc.chunk.chunk_id for rc in ev.chunks}
    assert "mu:risk" in ids  # the bridged neighbor's own filing — base never surfaced it
    assert "nvda:prov" in ids  # the edge's provenance chunk
    assert "nvda:1" in ids  # base recall preserved


def test_seed_from_query_when_no_filter() -> None:
    ev = _retriever().retrieve("What are NVIDIA's supply risks?", top_k=5)
    ids = {rc.chunk.chunk_id for rc in ev.chunks}
    assert "mu:risk" in ids  # seed resolved from the query text via the alias map


def test_no_seed_degrades_to_base() -> None:
    ev = _retriever().retrieve("generic question with no company", top_k=5)
    assert [rc.chunk.chunk_id for rc in ev.chunks] == ["nvda:1"]  # base verbatim


def test_no_edges_degrades_to_base() -> None:
    empty = _FakeGraph([], {})
    r = GraphRetriever(empty, _FakeBase(), _fetch, alias_map=_ALIAS)
    ev = r.retrieve("supply risk", top_k=5, where=ChunkFilter(ticker="NVDA"))
    assert [rc.chunk.chunk_id for rc in ev.chunks] == ["nvda:1"]


def test_union_dedups() -> None:
    ev = _retriever().retrieve("supply risk", top_k=10, where=ChunkFilter(ticker="NVDA"))
    ids = [rc.chunk.chunk_id for rc in ev.chunks]
    assert len(ids) == len(set(ids))  # no chunk appears twice after fusion


_NVDA_2 = _chunk("nvda:2", "NVDA also discloses demand risk.", "NVDA")


class _FullBase:
    """Base that fills top_k with NVDA chunks (nvda:1, nvda:2) + 1 MU chunk when scoped to MU."""

    name = "fullbase"

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        if where is not None and where.ticker == "MU":
            return EvidenceSet(query=query, chunks=[RetrievedChunk(chunk=_MU_RISK, score=0.7)])
        return EvidenceSet(query=query, chunks=[
            RetrievedChunk(chunk=_NVDA_BASE, score=0.9),  # nvda:1
            RetrievedChunk(chunk=_NVDA_2, score=0.8),     # nvda:2
        ])


def test_d1_supplier_not_starved_by_many_competitors() -> None:
    # D1 regression: a company with 5 competes_with edges (all conf 1.0) + 1 depends_on supplier,
    # max_neighbors=5. The OLD confidence-sorted-then-capped logic filled all 5 slots with
    # competitors (competes_with sorts before depends_on on ties) and dropped the supplier. The
    # round-robin must include the supplier (the bridge) even though competitors outnumber it.
    comps = ["AMD", "AVGO", "QCOM", "MRVL", "GOOGL"]
    edges = [
        Edge(subject="NVDA", relation="competes_with", object=t, provenance=[f"p:{t}"],
             filing_date=date(2025, 2, 26), source_url=f"https://sec.gov/{t}.htm", confidence=1.0)
        for t in comps
    ] + [
        Edge(subject="NVDA", relation="depends_on", object="MU", provenance=["p:MU"],
             filing_date=date(2025, 2, 26), source_url="https://sec.gov/MU.htm", confidence=1.0),
    ]
    ents = {t: Entity(id=t, name=t, type="company", ticker=t) for t in [*comps, "MU"]}
    gr = GraphRetriever(_FakeGraph(edges, ents), _FakeBase(), _fetch, alias_map=_ALIAS,
                        max_neighbors=5)
    picked = gr._neighbor_tickers(edges, exclude=set())
    assert "MU" in picked  # the depends_on supplier survives despite 5 competing edges
    assert len(picked) == 5  # still capped
    assert picked[0] == "MU"  # depends_on prioritized first in the round-robin


def test_d3_provenance_flood_does_not_crowd_out_neighbor() -> None:
    # D3 regression: an edge with MANY subject-own provenance chunks must NOT push the scoped
    # neighbor chunk below top_k. Scoped neighbor chunks lead the ranking; provenance is capped.
    prov_ids = [f"nvda:p{i}" for i in range(8)]
    edge = Edge(subject="NVDA", relation="depends_on", object="MU", provenance=prov_ids,
                filing_date=date(2025, 2, 26), source_url="https://sec.gov/N.htm", confidence=0.9)
    graph = _FakeGraph(
        [edge],
        {"MU": Entity(id="MU", name="Micron", type="company", ticker="MU"),
         "NVDA": Entity(id="NVDA", name="NVIDIA", type="company", ticker="NVDA")},
    )
    prov_chunks = {pid: _chunk(pid, "NVDA boilerplate provenance text", "NVDA") for pid in prov_ids}
    r = GraphRetriever(
        graph, _FakeBase(), lambda ids: {i: prov_chunks[i] for i in ids if i in prov_chunks},
        alias_map=_ALIAS,
    )
    ev = r.retrieve("supply risk", top_k=3, where=ChunkFilter(ticker="NVDA"))
    ids = [rc.chunk.chunk_id for rc in ev.chunks]
    assert "mu:risk" in ids  # neighbor payload survives 8 provenance chunks (which lead, pre-D3)


def test_f2_neighbor_surfaces_even_when_base_fills_topk() -> None:
    # F2 regression (adversarial): top_k=2, BOTH base chunks are also edge provenance. Without the
    # de-double-count, nvda:1 + nvda:2 each score from base AND graph → they take both slots and
    # the bridged MU chunk falls off. With F2 the graph list = {mu:risk} only, so MU surfaces.
    graph = _FakeGraph(
        [Edge(subject="NVDA", relation="depends_on", object="MU",
              provenance=["nvda:1", "nvda:2"],  # both are base chunks (the double-count trap)
              filing_date=date(2025, 2, 26), source_url="https://sec.gov/N.htm", confidence=0.9)],
        {"MU": Entity(id="MU", name="Micron", type="company", ticker="MU"),
         "NVDA": Entity(id="NVDA", name="NVIDIA", type="company", ticker="NVDA")},
    )
    table = {"nvda:1": _NVDA_BASE, "nvda:2": _NVDA_2, "mu:risk": _MU_RISK}
    r = GraphRetriever(
        graph, _FullBase(), lambda ids: {i: table[i] for i in ids if i in table}, alias_map=_ALIAS
    )
    ev = r.retrieve("supply risk", top_k=2, where=ChunkFilter(ticker="NVDA"))
    ids = [rc.chunk.chunk_id for rc in ev.chunks]
    assert "mu:risk" in ids  # neighbor evidence survived a full base ranking + the double-count
    assert len(ids) == len(set(ids)) == 2  # still exactly top_k, no dupes
