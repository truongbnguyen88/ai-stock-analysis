"""Retrieval-quality eval harness (RAG P9b; generalized in advanced-RAG A1).

Scores any ``RetrievalSystem`` (dense ``Retriever``, and — once they land — the A2 reranked, A3
hybrid, A5 graph variants) on a small **labeled** query set. Originally (P9b) this A/B'd
*embedders* (local ``fastembed`` vs ``voyage-4`` vs ``voyage-finance-2``) to *lock* the
production embedder + chunking before the one-time paid ingest (P9c) — embedding the wrong config
would burn the finite free token pool. A1 widens the unit-under-test from ``Embedder`` to
``RetrievalSystem`` and adds graded **nDCG@k** plus **citation accuracy**; ``run_ab`` stays as the
thin embedder-comparison wrapper over the generic ``evaluate_system``.

**Relevance is chunking-invariant by design.** A labeled query carries answer-bearing
``relevant_spans`` (phrases the right passage must contain); a retrieved chunk is relevant iff
its text contains *any* span (case-insensitive, whitespace-normalized). Because labels are
phrases rather than ``chunk_id``s, the *same* labels score different embedders **and** different
chunking configs — exactly what "confirm the chunking is settled" needs.

Pure + offline: the metrics are pure functions; ``run_ab`` embeds a fixed corpus with each
injected embedder into a fresh store and retrieves. Tests use ``FakeEmbedder`` +
``InMemoryVectorStore`` — no model download, no network. The real A/B (fastembed/voyage) is a
caller concern (keys/models); this module is provider-agnostic.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from stock_agent.rag.embeddings import Embedder
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.vector_store import InMemoryVectorStore, VectorStore
from stock_agent.schemas.documents import DocumentChunk, DocumentType
from stock_agent.schemas.research import GroundedAnswer
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace so span matching ignores casing/line-wrap noise."""
    return _WS.sub(" ", text).strip().lower()


@runtime_checkable
class RetrievalSystem(Protocol):
    """A named "query (+filter) -> ranked evidence" engine — the unit the harness scores.

    Every retrieval variant in the advanced track satisfies this one contract: the existing dense
    ``Retriever`` (A1), the A2 reranking wrapper, the A3 hybrid retriever, the A5 graph retriever.
    Because the metric code depends only on this Protocol — never a concrete class — adding a
    retrieval mode never touches eval, synthesis, memo, or the agent tools.
    """

    @property
    def name(self) -> str:
        """Identifier used in eval reports to distinguish retrieval systems (read-only)."""
        ...

    def retrieve(
        self, query: str, *, top_k: int, where: ChunkFilter | None = None
    ) -> EvidenceSet:
        """Return up to ``top_k`` ranked chunks for ``query`` (optionally scoped by ``where``)."""
        ...


class LabeledQuery(BaseModel):
    """One labeled retrieval example for the eval set.

    ``relevant_spans`` are answer-bearing phrases; a chunk's graded relevance is the number of
    DISTINCT spans it contains (chunking-invariant — see module docstring), gated by the optional
    metadata constraints below. ``ticker`` scopes retrieval AND the recall denominator. ``top_k``
    overrides the run-wide ``top_k`` for this query only.

    ``expected_document_types`` / ``expected_sections`` are optional, chunking-invariant
    constraints: a chunk only counts as relevant if its ``document_type`` / ``section`` is in the
    given set (e.g. require the answer to come from "Item 1A. Risk Factors" of a 10-K). Both
    default to ``None`` (no constraint), so legacy span-only labels are unchanged.
    """

    query: str
    relevant_spans: list[str] = Field(min_length=1)
    ticker: str | None = None
    top_k: int | None = None
    expected_document_types: list[DocumentType] | None = None
    expected_sections: list[str] | None = None

    def _passes_metadata(self, chunk: DocumentChunk) -> bool:
        """Whether ``chunk`` satisfies the optional doc-type / section constraints (AND)."""
        if (
            self.expected_document_types is not None
            and chunk.document_type not in self.expected_document_types
        ):
            return False
        return not (
            self.expected_sections is not None and chunk.section not in self.expected_sections
        )

    def relevance_grade(self, chunk: DocumentChunk) -> int:
        """Graded relevance: the count of DISTINCT answer spans ``chunk`` contains, or 0 if it
        fails any specified doc-type/section constraint. A chunk answering more of the question
        outranks one answering less — this integer is the nDCG gain (``rel_i``)."""
        if not self._passes_metadata(chunk):
            return 0
        haystack = _normalize(chunk.text)
        return sum(1 for span in self.relevant_spans if _normalize(span) in haystack)

    def is_relevant(self, chunk: DocumentChunk) -> bool:
        """Binary relevance (grade > 0): the chunk answers part of the query and passes metadata."""
        return self.relevance_grade(chunk) > 0


# ---- pure ranking metrics (operate on a ranked list of relevance flags) -------


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    """1.0 if any of the top-``k`` retrieved chunks is relevant, else 0.0 (a.k.a. success@k)."""
    return 1.0 if any(flags[:k]) else 0.0


def reciprocal_rank(flags: Sequence[bool], k: int) -> float:
    """``1/rank`` of the first relevant chunk within top-``k`` (0.0 if none). Mean = MRR."""
    for rank, flag in enumerate(flags[:k], start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    """Fraction of the *returned* top-``k`` that is relevant (denominator = chunks actually
    returned, so a corpus smaller than ``k`` is not penalized)."""
    top = flags[:k]
    return sum(top) / len(top) if top else 0.0


def recall_at_k(flags: Sequence[bool], n_relevant: int, k: int) -> float:
    """Fraction of the query's corpus-relevant chunks captured in top-``k``.

    ``n_relevant`` is the number of relevant chunks in the (ticker-scoped) corpus. Returns 0.0
    when ``n_relevant`` is 0 (a mislabeled query) — such queries are excluded from the mean.
    """
    if n_relevant <= 0:
        return 0.0
    return min(sum(flags[:k]), n_relevant) / n_relevant


def _dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain: ``sum_i (2^{g_i} - 1) / log2(i + 1)`` over 1-based ranks.

    The ``2^g - 1`` numerator (Järvelin & Kekäläinen 2002, the de-facto "exponential" DCG)
    rewards higher grades super-linearly; the ``log2(rank+1)`` denominator discounts later ranks.
    """
    # enumerate is 0-based, so position p maps to rank p+1 and discount log2(p+2).
    return float(sum((2.0**g - 1.0) / math.log2(p + 2) for p, g in enumerate(gains)))


def ndcg_at_k(
    gains: Sequence[float], k: int, *, ideal_gains: Sequence[float] | None = None
) -> float:
    """Normalized DCG@k in [0, 1] — graded, rank-discounted retrieval quality.

    ``gains`` are the per-rank relevance grades of the RETRIEVED list (top to bottom). The ideal
    DCG (denominator) is ``_dcg`` of ``ideal_gains`` sorted descending; pass the full set of
    corpus-relevant grades so relevant chunks that were NOT retrieved still inflate the ideal and
    correctly cap the score (otherwise nDCG is optimistic). Defaults to ``gains`` itself (the
    self-ideal textbook case). Returns 0.0 when IDCG is 0 (no relevant chunk exists).
    """
    dcg = _dcg(list(gains)[:k])
    pool = list(gains if ideal_gains is None else ideal_gains)
    idcg = _dcg(sorted(pool, reverse=True)[:k])
    return dcg / idcg if idcg > 0 else 0.0


def citation_accuracy(
    answer: GroundedAnswer, evidence: EvidenceSet, labeled: LabeledQuery
) -> float | None:
    """Precision of an answer's citations: fraction that point to a RELEVANT retrieved chunk.

    For each inline ``[n]`` citation we resolve its ``chunk_id`` within the retrieved ``evidence``
    and test that chunk against the labeled answer spans. A citation to a chunk that was never
    retrieved counts as wrong (the P7 citation guard should already preclude it). Returns ``None``
    when the answer makes no citations (an honest refusal / no claim) so the aggregate can EXCLUDE
    it rather than score a non-answer 0 — mirroring how recall excludes mislabeled queries.

    NB: this is the *deterministic* generation-quality metric (citation precision); the softer,
    paid **faithfulness** judge (is each claim entailed by its cited chunk?) is a separate opt-in
    layer, deferred — the hard floor (no invented citations/numbers) is the P7 guards' job.
    """
    if not answer.citations:
        return None
    by_id = {rc.chunk.chunk_id: rc.chunk for rc in evidence.chunks}
    good = sum(
        1
        for c in answer.citations
        if (chunk := by_id.get(c.chunk_id)) is not None and labeled.is_relevant(chunk)
    )
    return good / len(answer.citations)


class QueryReport(BaseModel):
    """Per-query retrieval metrics for one retrieval system."""

    query: str
    k: int
    n_relevant_corpus: int  # relevant chunks present in the (scoped) corpus — recall denominator
    n_retrieved: int
    hit: float
    reciprocal_rank: float
    precision: float
    recall: float
    ndcg: float


class SystemReport(BaseModel):
    """Aggregate (mean) retrieval metrics for one ``RetrievalSystem`` across the labeled set."""

    system: str
    top_k: int
    n_queries: int
    hit_at_k: float
    mrr: float
    ndcg_at_k: float
    precision_at_k: float
    recall_at_k: float
    per_query: list[QueryReport]


class EmbedderReport(SystemReport):
    """Back-compat alias (P9b): an embedder A/B is just a comparison of *dense* systems, so it is a
    ``SystemReport`` whose ``system`` is the embedder name. ``embedder`` mirrors ``system`` for
    callers/tests written against the original P9b field name."""

    @property
    def embedder(self) -> str:
        return self.system


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_query(
    system: RetrievalSystem,
    labeled: LabeledQuery,
    *,
    top_k: int,
    corpus_chunks: Sequence[DocumentChunk],
) -> QueryReport:
    """Retrieve ``labeled`` with ``system`` and score it against the answer spans.

    ``corpus_chunks`` is the full ingested corpus; its (ticker-scoped) relevance *grades* give
    both the recall denominator (count of grade>0 chunks) and the ideal-DCG pool for nDCG — so a
    relevant chunk that was not retrieved correctly caps nDCG. Retrieval is scoped by
    ``labeled.ticker`` (the same filter used in prod).
    """
    k = labeled.top_k or top_k
    where = ChunkFilter(ticker=labeled.ticker) if labeled.ticker else None
    evidence = system.retrieve(labeled.query, top_k=k, where=where)
    flags = [labeled.is_relevant(rc.chunk) for rc in evidence.chunks]
    retrieved_gains = [float(labeled.relevance_grade(rc.chunk)) for rc in evidence.chunks]
    corpus_gains = [
        float(labeled.relevance_grade(c))
        for c in corpus_chunks
        if labeled.ticker is None or c.ticker == labeled.ticker
    ]
    n_relevant = sum(1 for g in corpus_gains if g > 0)
    return QueryReport(
        query=labeled.query,
        k=k,
        n_relevant_corpus=n_relevant,
        n_retrieved=len(flags),
        hit=hit_at_k(flags, k),
        reciprocal_rank=reciprocal_rank(flags, k),
        precision=precision_at_k(flags, k),
        recall=recall_at_k(flags, n_relevant, k),
        ndcg=ndcg_at_k(retrieved_gains, k, ideal_gains=corpus_gains),
    )


def evaluate_system(
    system: RetrievalSystem,
    queries: Sequence[LabeledQuery],
    *,
    top_k: int = 8,
    corpus_chunks: Sequence[DocumentChunk],
) -> SystemReport:
    """Score one ``RetrievalSystem`` over the labeled set → aggregate (mean) ``SystemReport``.

    The generic unit-under-test for the whole advanced track: dense (A1), reranked (A2), hybrid
    (A3), graph (A5) all plug in here unchanged. ``ndcg_at_k`` and ``recall_at_k`` means EXCLUDE
    queries with no corpus-relevant chunk (a mislabeled / out-of-corpus query would otherwise drag
    the mean toward 0); ``hit``/``mrr``/``precision`` average over all queries.
    """
    per_query = [
        evaluate_query(system, q, top_k=top_k, corpus_chunks=corpus_chunks) for q in queries
    ]
    return SystemReport(
        system=system.name,
        top_k=top_k,
        n_queries=len(per_query),
        hit_at_k=_mean([q.hit for q in per_query]),
        mrr=_mean([q.reciprocal_rank for q in per_query]),
        ndcg_at_k=_mean([q.ndcg for q in per_query if q.n_relevant_corpus > 0]),
        precision_at_k=_mean([q.precision for q in per_query]),
        recall_at_k=_mean([q.recall for q in per_query if q.n_relevant_corpus > 0]),
        per_query=per_query,
    )


def run_ab(
    corpus_chunks: Sequence[DocumentChunk],
    queries: Sequence[LabeledQuery],
    embedders: Mapping[str, Embedder],
    *,
    top_k: int = 8,
    store_factory: Callable[[], VectorStore] = InMemoryVectorStore,
) -> list[EmbedderReport]:
    """Embed ``corpus_chunks`` with each embedder into a fresh store and score every query.

    Chunking is held constant across embedders (the corpus is chunked once by the caller); to
    A/B chunking, run twice with different ``corpus_chunks``. Returns one report per embedder,
    in ``embedders`` iteration order. ``recall_at_k`` averages only queries with >0 relevant
    chunks (mislabeled / out-of-corpus queries don't distort it).
    """
    reports: list[EmbedderReport] = []
    for name, embedder in embedders.items():
        store = store_factory()
        if corpus_chunks:
            vectors = embedder.embed_documents([c.text for c in corpus_chunks])
            store.add(list(corpus_chunks), vectors)
        retriever = Retriever(embedder, store, name=name)
        report = evaluate_system(retriever, queries, top_k=top_k, corpus_chunks=corpus_chunks)
        # Same fields as SystemReport; re-wrap so the back-compat `.embedder` accessor is present.
        reports.append(EmbedderReport.model_validate(report.model_dump()))
    return reports


def format_reports_markdown(reports: Sequence[SystemReport]) -> str:
    """Render the reports as a GitHub-flavored Markdown comparison table (CLI output).

    Generic over any ``SystemReport`` (dense embedder A/B, or later dense-vs-hybrid-vs-reranked),
    so the same renderer serves the whole advanced track.
    """
    if not reports:
        return "_no embedders evaluated_"
    k = reports[0].top_k
    lines = [
        f"| system | hit@{k} | MRR | nDCG@{k} | precision@{k} | recall@{k} | queries |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.system} | {r.hit_at_k:.3f} | {r.mrr:.3f} | {r.ndcg_at_k:.3f} | "
            f"{r.precision_at_k:.3f} | {r.recall_at_k:.3f} | {r.n_queries} |"
        )
    return "\n".join(lines)
