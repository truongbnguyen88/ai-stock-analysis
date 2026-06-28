"""Graph retrieval (advanced-RAG A5.2) — graph traversal as a drop-in ``RetrievalSystem``.

``GraphRetriever`` answers a query by combining vector retrieval with a graph **traversal**:

1. **base** — the wrapped vector/hybrid retriever runs as usual (the recall floor).
2. **seed** — resolve the query's anchor entity (``where.ticker`` or alias-matched companies).
3. **traverse** — BFS the graph from each seed → the edges crossed give (a) **provenance** chunk_ids
   (the chunks the relationships were *stated* in — the "who") and (b) **neighbor** companies.
4. **scope** — for each neighbor company, run a vector retrieval of the SAME query scoped to that
   company (the "what" — e.g. the neighbor's own risk disclosure that a query about the seed never
   surfaces; the bridging hop §16.7).
5. **fuse** — combine the base ranking with the graph-sourced ranking by **Reciprocal Rank Fusion**
   (the same rank-based fusion as A3 hybrid), so graph evidence surfaces regardless of the
   incomparable score scales (cosine vs. confidence vs. RRF), then take ``top_k``.

It satisfies the ``RetrievalSystem`` contract (``name`` + ``retrieve``), so it drops into
``build_retrieval_system`` / the A4 loop / the eval harness with **zero** caller changes. It is
$0/local (no LLM). Grounding is unchanged: it only *selects* chunk_ids — the terminal P7 synthesis
runs the citation + number guards over the union exactly as before. With no seed or no edges it
returns the base result verbatim (graph never *loses* recall, it only adds).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from stock_agent.graph.store import GraphStore
from stock_agent.logging_config import get_logger
from stock_agent.rag.hybrid import reciprocal_rank_fusion
from stock_agent.rag.retriever import RetrievalSystem
from stock_agent.research.bridge import load_alias_map, mentioned_tickers
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.graph import Edge, Relation
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk

log = get_logger(__name__)

# Materializes provenance chunks by id (prod: the sparse store's fetch; tests inject a fake).
ChunkFetch = Callable[[Sequence[str]], dict[str, DocumentChunk]]

# Round-robin order for balanced neighbor selection (D1). depends_on first — it is the canonical
# bridge relation (supplier/foundry dependency, §A5); competes_with second. risk/topic relations
# never yield company tickers, so they don't contribute neighbors, but are listed for completeness.
_RELATION_PRIORITY: tuple[Relation, ...] = (
    "depends_on", "competes_with", "exposed_to", "mentions_risk",
)


class GraphRetriever:
    """Vector retrieval ⊕ graph traversal, fused by RRF; a drop-in ``RetrievalSystem``.

    ``base`` is the wrapped retriever (dense/hybrid); ``graph`` is the knowledge graph;
    ``chunk_fetch`` materializes provenance chunks by id. ``relations`` restricts edge types
    (``None`` = all four); ``hops`` is the traversal depth; ``max_neighbors`` caps the scoped per-
    neighbor searches; ``per_neighbor_k`` is each scoped search's top-k. ``alias_map`` (for
    tests) seeds entity resolution from the query when no ticker filter is given.
    """

    def __init__(
        self,
        graph: GraphStore,
        base: RetrievalSystem,
        chunk_fetch: ChunkFetch,
        *,
        alias_map: Mapping[str, Sequence[str]] | None = None,
        relations: Sequence[Relation] | None = None,
        hops: int = 1,
        max_neighbors: int = 5,
        per_neighbor_k: int = 4,
        max_provenance: int = 4,
        k_rrf: int = 60,
    ) -> None:
        self._graph = graph
        self._base = base
        self._chunk_fetch = chunk_fetch
        self._alias_map = alias_map if alias_map is not None else load_alias_map()
        self._relations = relations
        self._hops = hops
        self._max_neighbors = max_neighbors
        self._per_neighbor_k = per_neighbor_k
        self._max_provenance = max_provenance  # cap subject-own provenance chunks (D3 anti-flood)
        self._k_rrf = k_rrf

    @property
    def name(self) -> str:
        return f"graph({self._base.name})"

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        """Return up to ``top_k`` chunks fusing base retrieval with graph-traversed evidence."""
        base_ev = self._base.retrieve(query, top_k=top_k, where=where)
        seeds = self._seeds(query, where)
        if not seeds:
            return base_ev  # nothing to anchor the traversal → identical to base

        graph_chunks = self._traverse(query, seeds)
        if not graph_chunks:
            return base_ev  # seed(s) have no edges in the graph → identical to base

        # Fuse the two rankings by RRF (rank-based → no cosine/confidence/BM25 normalization).
        # F2 — de-double-count: a chunk already in the base ranking must NOT also score from the
        # graph list. Edge-provenance chunks are usually the seed's OWN chunks (already in base), so
        # without this they'd be counted twice and dominate the fusion, pushing the genuinely-new
        # bridged-neighbor chunks below top_k (bug: MU's risk chunk was gathered, never surfaced).
        # The graph list thus contributes only evidence the base didn't already return, so neighbor
        # chunks rank on their own merit and reliably appear in the result.
        base_ids = [rc.chunk.chunk_id for rc in base_ev.chunks]
        base_set = set(base_ids)
        graph_ids = [
            rc.chunk.chunk_id for rc in graph_chunks if rc.chunk.chunk_id not in base_set
        ]
        fused = reciprocal_rank_fusion([base_ids, graph_ids], k=self._k_rrf)[:top_k]

        by_id = {rc.chunk.chunk_id: rc.chunk for rc in base_ev.chunks}
        for rc in graph_chunks:  # add graph-only chunks (base keeps priority on shared ids)
            by_id.setdefault(rc.chunk.chunk_id, rc.chunk)
        chunks = [
            RetrievedChunk(chunk=by_id[cid], score=score)
            for cid, score in fused
            if cid in by_id  # defensive: an id neither source can materialize is skipped
        ]
        return EvidenceSet(query=query, chunks=chunks)

    def _seeds(self, query: str, where: ChunkFilter | None) -> list[str]:
        """Anchor entities: the filter's ticker, else companies named in the query."""
        if where is not None and where.ticker:
            return [where.ticker]
        return sorted(mentioned_tickers(query, self._alias_map))

    def _traverse(self, query: str, seeds: Sequence[str]) -> list[RetrievedChunk]:
        """Graph-sourced evidence for ``seeds``, ordered so the BRIDGE PAYLOAD leads the ranking.

        Returns the **scoped neighbor chunks first** (the "what" — each neighbor company's own
        filings retrieved for the query) followed by a **capped** set of **provenance** chunks (the
        "who" — where the edge was stated). D3 fix: provenance used to lead, and for a hub like NVDA
        with dozens of edges that meant dozens of the *subject's own* provenance chunks flooded the
        top of the graph ranking and pushed the neighbor payload below ``top_k`` (so the bridge was
        gathered but never surfaced). Neighbor chunks are the value-add, so they rank first;
        provenance is supporting and capped at ``max_provenance`` per seed (usually already in
        ``base`` anyway when the query is seed-scoped). Order here = the RRF rank in ``retrieve``.
        """
        scoped: list[RetrievedChunk] = []
        provenance: list[RetrievedChunk] = []
        seen_ids: set[str] = set()
        seed_set = set(seeds)
        for seed in seeds:
            edges = self._graph.neighbors(seed, relations=self._relations, hops=self._hops)
            if not edges:
                continue
            # (what) scoped vector retrieval per neighbor company — the bridge payload, leads.
            for ticker in self._neighbor_tickers(edges, exclude=seed_set):
                ev = self._base.retrieve(
                    query, top_k=self._per_neighbor_k, where=ChunkFilter(ticker=ticker)
                )
                for rc in ev.chunks:
                    if rc.chunk.chunk_id not in seen_ids:
                        seen_ids.add(rc.chunk.chunk_id)
                        scoped.append(rc)
            # (who) provenance chunks (highest-confidence first), CAPPED — supporting evidence only.
            conf_by_id: dict[str, float] = {}
            order: list[str] = []
            for edge in sorted(edges, key=lambda e: -e.confidence):
                for cid in edge.provenance:
                    if cid not in conf_by_id:
                        conf_by_id[cid] = edge.confidence
                        order.append(cid)
            order = order[: self._max_provenance]
            fetched = self._chunk_fetch(order)
            for cid in order:
                if cid in fetched and cid not in seen_ids:
                    seen_ids.add(cid)
                    provenance.append(RetrievedChunk(chunk=fetched[cid], score=conf_by_id[cid]))
        return scoped + provenance

    def _neighbor_tickers(self, edges: Sequence[Edge], exclude: set[str]) -> list[str]:
        """Distinct neighbor company tickers, **balanced across relations** (round-robin), capped.

        D1 fix: a single confidence-sorted list let one relation starve another — a company with
        many ``competes_with`` edges (all at confidence 1.0) filled ``max_neighbors`` before any
        ``depends_on`` supplier was reached (ties broke by edge-key, and "competes_with" sorts
        before "depends_on"), so the supplier bridge (e.g. NVDA→MU) was silently dropped. Instead we
        build a per-relation queue (best confidence first, ties by ticker) and round-robin across
        relations in a fixed priority order, so suppliers and competitors both get representation
        regardless of how lopsided the edge counts are. Only company nodes with a resolved ticker
        are scoped-searched (risk/topic nodes have none); seeds are excluded (base covers them).
        """
        scored: dict[Relation, dict[str, float]] = {}
        for edge in edges:
            ent = self._graph.get_entity(edge.object)
            if not (ent and ent.type == "company" and ent.ticker and ent.ticker not in exclude):
                continue
            best = scored.setdefault(edge.relation, {})
            best[ent.ticker] = max(best.get(ent.ticker, 0.0), edge.confidence)
        # Per-relation queues, highest confidence first (ties by ticker for determinism).
        queues = {
            rel: [t for t, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]
            for rel, best in scored.items()
        }
        order = [r for r in _RELATION_PRIORITY if r in queues]

        out: list[str] = []
        seen = set(exclude)
        cursor = dict.fromkeys(order, 0)
        while len(out) < self._max_neighbors:
            progressed = False
            for rel in order:
                q, i = queues[rel], cursor[rel]
                while i < len(q) and q[i] in seen:  # skip tickers already taken by another relation
                    i += 1
                cursor[rel] = i
                if i < len(q):
                    seen.add(q[i])
                    out.append(q[i])
                    cursor[rel] = i + 1
                    progressed = True
                    if len(out) >= self._max_neighbors:
                        break
            if not progressed:  # every queue exhausted
                break
        return out
