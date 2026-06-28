"""Graph store (advanced-RAG A5.0) — SqliteGraphStore writes, idempotency, and BFS traversal.

Hand-built triples in a temp SQLite DB (no LLM, no network): add → 1-/2-hop neighbors, directional
traversal, idempotent upsert, provenance retrieval, cycle termination, and namespacing.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from stock_agent.graph.store import GraphStore, SqliteGraphStore, build_graph_store
from stock_agent.schemas.graph import Edge, Entity
from stock_agent.settings import Settings

_URL = "https://www.sec.gov/Archives/edgar/data/x/nvda.htm"


def _edge(
    subj: str, rel: str, obj: str, *, prov: list[str], url: str = _URL, conf: float = 0.9
) -> Edge:
    return Edge(
        subject=subj, relation=rel, object=obj, provenance=prov,
        filing_date=date(2025, 2, 26), source_url=url, confidence=conf,
    )


def _company(tic: str, name: str) -> Entity:
    return Entity(id=tic, name=name, type="company", ticker=tic)


def _store(tmp_path: Path) -> SqliteGraphStore:
    return SqliteGraphStore(tmp_path / "graph.db")


def test_protocol_conformance(tmp_path: Path) -> None:
    assert isinstance(_store(tmp_path), GraphStore)


def test_add_and_get_entity(tmp_path: Path) -> None:
    g = _store(tmp_path)
    g.add_entities(
        [_company("NVDA", "NVIDIA"), Entity(id="capacity", name="capacity", type="risk")]
    )
    assert g.get_entity("NVDA") == _company("NVDA", "NVIDIA")
    cap = g.get_entity("capacity")
    assert cap is not None and cap.ticker is None  # non-company node carries no ticker
    assert g.get_entity("MISSING") is None


def test_one_hop_neighbors_with_provenance(tmp_path: Path) -> None:
    g = _store(tmp_path)
    g.add_entities([_company("NVDA", "NVIDIA"), _company("MU", "Micron")])
    g.add_edges([_edge("NVDA", "depends_on", "MU", prov=["NVDA:10-K:2025-02-26:7"])])
    edges = g.neighbors("NVDA", hops=1)
    assert [e.object for e in edges] == ["MU"]
    assert edges[0].provenance == ["NVDA:10-K:2025-02-26:7"]
    assert g.provenance_chunk_ids("NVDA") == ["NVDA:10-K:2025-02-26:7"]


def test_two_hop_traversal(tmp_path: Path) -> None:
    g = _store(tmp_path)
    g.add_edges(
        [
            _edge("NVDA", "depends_on", "MU", prov=["c1"]),
            _edge("MU", "depends_on", "ASML", prov=["c2"]),
        ]
    )
    one = {e.object for e in g.neighbors("NVDA", hops=1)}
    two = {e.object for e in g.neighbors("NVDA", hops=2)}
    assert one == {"MU"}  # ASML is 2 hops away, not reached at depth 1
    assert two == {"MU", "ASML"}
    # provenance follows neighbors()' deterministic edge-key order (MU<NVDA on subject) → [c2, c1]
    assert g.provenance_chunk_ids("NVDA", hops=2) == ["c2", "c1"]


def test_direction_in_and_both(tmp_path: Path) -> None:
    g = _store(tmp_path)
    g.add_edges([_edge("NVDA", "depends_on", "MU", prov=["c1"])])
    # Out from NVDA reaches MU; out from MU reaches nothing; IN to MU reaches NVDA (inverse hop).
    assert {e.object for e in g.neighbors("NVDA", direction="out")} == {"MU"}
    assert g.neighbors("MU", direction="out") == []
    in_edges = g.neighbors("MU", direction="in")
    assert [e.subject for e in in_edges] == ["NVDA"]
    assert len(g.neighbors("MU", direction="both")) == 1


def test_relation_filter(tmp_path: Path) -> None:
    g = _store(tmp_path)
    g.add_edges(
        [
            _edge("NVDA", "depends_on", "MU", prov=["c1"]),
            _edge("NVDA", "competes_with", "AMD", prov=["c2"]),
        ]
    )
    deps = g.neighbors("NVDA", relations=["depends_on"])
    assert [e.object for e in deps] == ["MU"]


def test_idempotent_upsert(tmp_path: Path) -> None:
    g = _store(tmp_path)
    e = _edge("NVDA", "depends_on", "MU", prov=["c1"], conf=0.5)
    g.add_entities([_company("NVDA", "NVIDIA")])
    g.add_edges([e])
    g.add_entities([_company("NVDA", "NVIDIA")])  # same node again
    g.add_edges([_edge("NVDA", "depends_on", "MU", prov=["c1", "c9"], conf=0.95)])  # re-extract
    counts = g.count()
    assert counts.nodes == 1 and counts.edges == 1  # no duplication
    # latest write wins: provenance + confidence updated in place
    edge = g.neighbors("NVDA")[0]
    assert edge.provenance == ["c1", "c9"]
    assert edge.confidence == 0.95


def test_same_triple_different_filing_is_two_edges(tmp_path: Path) -> None:
    g = _store(tmp_path)
    g.add_edges(
        [
            _edge("NVDA", "depends_on", "MU", prov=["c1"], url="https://sec.gov/2024.htm"),
            _edge("NVDA", "depends_on", "MU", prov=["c2"], url="https://sec.gov/2025.htm"),
        ]
    )
    assert g.count().edges == 2  # distinct source_url → distinct evidence


def test_cycle_terminates(tmp_path: Path) -> None:
    g = _store(tmp_path)
    g.add_edges(
        [
            _edge("NVDA", "competes_with", "AMD", prov=["c1"]),
            _edge("AMD", "competes_with", "NVDA", prov=["c2"]),
        ]
    )
    edges = g.neighbors("NVDA", hops=5, direction="both")  # would loop without visited-set
    assert len(edges) == 2


def test_namespacing(tmp_path: Path) -> None:
    settings = Settings(graph_store_dir=tmp_path)
    g = build_graph_store(settings)
    g.add_entities([_company("NVDA", "NVIDIA")])
    assert g.count().nodes == 1
    assert list(tmp_path.glob("*.db"))  # a namespaced .db file was created
