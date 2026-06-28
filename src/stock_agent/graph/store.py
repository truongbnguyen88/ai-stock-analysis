"""Graph store (advanced-RAG A5.0) — a persistent SQLite knowledge graph + bounded BFS traversal.

The source of truth for the A5 entity–relationship graph: two tables (``nodes`` / ``edges``) with
**idempotent upserts** and a small traversal API. Traversal is the whole point — ``neighbors`` runs
a bounded breadth-first search (1–2 hops; the matrix view is the walk-count sum over hops, the
implementation is BFS over the reachable subgraph, see docs/rag_concepts.md §16.2) and returns the
**edges** it crossed, so the caller gets both the neighbor entities (``edge.object``) and the
provenance ``chunk_id``s (``edge.provenance``) needed to ground a graph answer.

No LLM, no network — pure storage + graph algorithms (A5.0 is the queryable substrate; extraction is
A5.1, retrieval is A5.2). NetworkX is deliberately **not** used: our per-ticker graphs are tiny, BFS
over indexed SQLite rows is exact, dependency-free, and trivially debuggable.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from stock_agent.logging_config import get_logger
from stock_agent.schemas.graph import Edge, Entity, Relation
from stock_agent.settings import Settings

log = get_logger(__name__)

Direction = Literal["out", "in", "both"]


@dataclass(frozen=True)
class GraphCounts:
    """Node/edge cardinality of a graph store (for status + tests)."""

    nodes: int
    edges: int


def _edge_key(e: Edge) -> tuple[str, str, str, str]:
    """Stable identity of an edge: the triple + the filing it was stated in.

    The same ``(subject, relation, object)`` asserted by two *different* filings (distinct
    ``source_url``) is two pieces of evidence kept as two rows; re-extracting the *same* filing
    upserts the one row (idempotent). This is the edge analogue of chunk-id stability.
    """
    return (e.subject, e.relation, e.object, e.source_url)


@runtime_checkable
class GraphStore(Protocol):
    """A typed knowledge graph: idempotent writes + bounded neighbor traversal.

    Implementations persist ``Entity`` nodes and provenance-bearing ``Edge`` triples and expose a
    k-hop ``neighbors`` traversal. The A5.2 ``GraphRetriever`` depends only on this Protocol.
    """

    def add_entities(self, entities: Sequence[Entity]) -> None:
        """Upsert ``entities`` by ``id`` (idempotent)."""
        ...

    def add_edges(self, edges: Sequence[Edge]) -> None:
        """Upsert ``edges`` by ``(subject, relation, object, source_url)`` (idempotent)."""
        ...

    def get_entity(self, entity_id: str) -> Entity | None:
        """The entity with this id, or ``None``."""
        ...

    def neighbors(
        self,
        entity_id: str,
        *,
        relations: Sequence[Relation] | None = None,
        hops: int = 1,
        direction: Direction = "out",
    ) -> list[Edge]:
        """Edges crossed by a bounded BFS from ``entity_id`` (≤ ``hops`` levels), deduped."""
        ...

    def provenance_chunk_ids(
        self,
        entity_id: str,
        *,
        relations: Sequence[Relation] | None = None,
        hops: int = 1,
        direction: Direction = "out",
    ) -> list[str]:
        """Union of provenance ``chunk_id``s across the traversed edges (first-seen order)."""
        ...

    def count(self) -> GraphCounts:
        """Node + edge cardinality."""
        ...


class SqliteGraphStore:
    """Persistent knowledge graph in SQLite: ``nodes`` + ``edges`` tables, BFS traversal.

    Writes are upserts (``ON CONFLICT … DO UPDATE``) keyed on the entity id / the edge identity
    (``_edge_key``), so re-running extraction never duplicates. ``edges.provenance`` is stored as a
    JSON array of ``chunk_id``s. Indices on ``subject`` and ``object`` make both out- and in-edge
    lookups O(degree). ``check_same_thread=False`` mirrors the sparse store (the agent may query
    from a worker thread; access here is single-connection and serialized).
    """

    name = "sqlite"

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS nodes ("
            "id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, ticker TEXT)"
        )
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS edges ("
            "subject TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL, "
            "provenance TEXT NOT NULL, filing_date TEXT NOT NULL, source_url TEXT NOT NULL, "
            "confidence REAL NOT NULL, "
            "PRIMARY KEY (subject, relation, object, source_url))"
        )
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_edges_subject ON edges(subject)")
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_edges_object ON edges(object)")
        self._con.commit()

    # ------------------------------------------------------------------ writes
    def add_entities(self, entities: Sequence[Entity]) -> None:
        """Upsert ``entities`` by id (latest write wins on name/type/ticker)."""
        if not entities:
            return
        self._con.executemany(
            "INSERT INTO nodes (id, name, type, ticker) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET name=excluded.name, type=excluded.type, "
            "ticker=excluded.ticker",
            [(e.id, e.name, e.type, e.ticker) for e in entities],
        )
        self._con.commit()

    def add_edges(self, edges: Sequence[Edge]) -> None:
        """Upsert ``edges`` by ``_edge_key`` (provenance/date/confidence replaced on re-extract)."""
        if not edges:
            return
        self._con.executemany(
            "INSERT INTO edges "
            "(subject, relation, object, provenance, filing_date, source_url, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(subject, relation, object, source_url) DO UPDATE SET "
            "provenance=excluded.provenance, filing_date=excluded.filing_date, "
            "confidence=excluded.confidence",
            [
                (
                    e.subject, e.relation, e.object, json.dumps(e.provenance),
                    e.filing_date.isoformat(), e.source_url, e.confidence,
                )
                for e in edges
            ],
        )
        self._con.commit()

    # ------------------------------------------------------------------ reads
    def get_entity(self, entity_id: str) -> Entity | None:
        """The entity row as an ``Entity``, or ``None`` if absent."""
        row = self._con.execute(
            "SELECT id, name, type, ticker FROM nodes WHERE id = ?", (entity_id,)
        ).fetchone()
        if row is None:
            return None
        return Entity(id=row[0], name=row[1], type=row[2], ticker=row[3])

    def _adjacent(
        self, node: str, relations: Sequence[Relation] | None, direction: Direction
    ) -> list[Edge]:
        """Edges incident to ``node`` in the requested ``direction`` (optionally relation-filtered).

        ``out`` = edges with ``subject == node``; ``in`` = edges with ``object == node``; ``both`` =
        the union. Relation filtering is applied in SQL. Results are ordered deterministically so
        the traversal (and every downstream union) is reproducible.
        """
        clauses: list[str] = []
        if direction == "out":
            clauses.append("subject = ?")
            params: list[str] = [node]
        elif direction == "in":
            clauses.append("object = ?")
            params = [node]
        else:  # both
            clauses.append("(subject = ? OR object = ?)")
            params = [node, node]
        rel_filter = ""
        rel_params: list[str] = []
        if relations:
            rel_filter = f" AND relation IN ({','.join('?' for _ in relations)})"
            rel_params = list(relations)
        sql = (
            "SELECT subject, relation, object, provenance, filing_date, source_url, confidence "
            f"FROM edges WHERE {clauses[0]}{rel_filter} "
            "ORDER BY subject, relation, object, source_url"
        )
        rows = self._con.execute(sql, (*params, *rel_params)).fetchall()
        return [self._row_to_edge(r) for r in rows]

    @staticmethod
    def _row_to_edge(row: tuple[str, str, str, str, str, str, float]) -> Edge:
        """Reconstruct an ``Edge`` from a DB row (parse the JSON provenance + ISO date)."""
        return Edge(
            subject=row[0],
            relation=row[1],  # constrained to Relation on write; pydantic validates on read
            object=row[2],
            provenance=json.loads(row[3]),
            filing_date=Date.fromisoformat(row[4]),
            source_url=row[5],
            confidence=row[6],
        )

    def neighbors(
        self,
        entity_id: str,
        *,
        relations: Sequence[Relation] | None = None,
        hops: int = 1,
        direction: Direction = "out",
    ) -> list[Edge]:
        """Bounded BFS from ``entity_id``: the deduped edges crossed within ``hops`` levels.

        Each level expands the frontier to the *other* endpoint of every incident edge (``object``
        for an out-edge, ``subject`` for an in-edge), never revisiting a node, so a cyclic graph
        (e.g. mutual ``competes_with``) terminates. Returns edges (not just neighbor ids) because
        the retriever needs both the neighbor (``edge.object``) and provenance (``edge.provenance``)
        to ground the answer.
        ``hops <= 0`` returns ``[]``.
        """
        if hops <= 0:
            return []
        visited: set[str] = {entity_id}
        frontier: set[str] = {entity_id}
        collected: dict[tuple[str, str, str, str], Edge] = {}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in sorted(frontier):  # sorted → deterministic expansion order
                for edge in self._adjacent(node, relations, direction):
                    collected[_edge_key(edge)] = edge
                    other = edge.object if edge.subject == node else edge.subject
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
            frontier = next_frontier
            if not frontier:
                break
        return sorted(collected.values(), key=_edge_key)

    def provenance_chunk_ids(
        self,
        entity_id: str,
        *,
        relations: Sequence[Relation] | None = None,
        hops: int = 1,
        direction: Direction = "out",
    ) -> list[str]:
        """Union of provenance chunk ids across the traversed edges (deduped).

        Order follows ``neighbors`` (deterministic edge-key order), so the retriever materializes a
        reproducible chunk set; it re-scores/re-ranks anyway, so exact order is not load-bearing.
        """
        seen: set[str] = set()
        out: list[str] = []
        for edge in self.neighbors(
            entity_id, relations=relations, hops=hops, direction=direction
        ):
            for cid in edge.provenance:
                if cid not in seen:
                    seen.add(cid)
                    out.append(cid)
        return out

    def count(self) -> GraphCounts:
        """Node + edge cardinality (one query each)."""
        nodes = self._con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        edges = self._con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        return GraphCounts(nodes=nodes, edges=edges)


def _slug(text: str) -> str:
    """Filesystem-safe slug of an embedder namespace (mirrors the vector/sparse stores)."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()


def build_graph_store(settings: Settings) -> SqliteGraphStore:
    """Build the persistent graph store, namespaced per embedder (parallel to the other stores).

    Namespacing keeps graphs built over corpora chunked by different embedder configs from mixing
    (the graph's chunk-id provenance must resolve in the matching vector/sparse store).
    """
    from stock_agent.rag.embeddings import embedding_namespace

    return SqliteGraphStore(settings.graph_store_dir / f"{_slug(embedding_namespace(settings))}.db")
