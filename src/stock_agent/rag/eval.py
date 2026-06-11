"""Retrieval-quality A/B harness (RAG P9b).

Compares embedders (e.g. local ``fastembed`` vs ``voyage-4`` vs ``voyage-finance-2``) on a
small **labeled** query set so the production embedder + chunking can be *locked before* the
one-time paid ingest (P9c) — embedding the wrong config would burn the finite free token pool.

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

import re
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, Field

from stock_agent.rag.embeddings import Embedder
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.vector_store import InMemoryVectorStore, VectorStore
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter

_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace so span matching ignores casing/line-wrap noise."""
    return _WS.sub(" ", text).strip().lower()


class LabeledQuery(BaseModel):
    """One labeled retrieval example for the A/B set.

    ``relevant_spans`` are answer-bearing phrases; a chunk is relevant iff it contains any of
    them (chunking-invariant — see module docstring). ``ticker`` scopes retrieval AND the
    recall denominator. ``top_k`` overrides the run-wide ``top_k`` for this query only.
    """

    query: str
    relevant_spans: list[str] = Field(min_length=1)
    ticker: str | None = None
    top_k: int | None = None

    def is_relevant(self, chunk: DocumentChunk) -> bool:
        """Whether ``chunk`` contains an answer span (case-insensitive, whitespace-normalized)."""
        haystack = _normalize(chunk.text)
        return any(_normalize(span) in haystack for span in self.relevant_spans)


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


class QueryReport(BaseModel):
    """Per-query retrieval metrics for one embedder."""

    query: str
    k: int
    n_relevant_corpus: int  # relevant chunks present in the (scoped) corpus — recall denominator
    n_retrieved: int
    hit: float
    reciprocal_rank: float
    precision: float
    recall: float


class EmbedderReport(BaseModel):
    """Aggregate (mean) retrieval metrics for one embedder across the labeled set."""

    embedder: str
    top_k: int
    n_queries: int
    hit_at_k: float
    mrr: float
    precision_at_k: float
    recall_at_k: float
    per_query: list[QueryReport]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_query(
    retriever: Retriever,
    labeled: LabeledQuery,
    *,
    top_k: int,
    corpus_chunks: Sequence[DocumentChunk],
) -> QueryReport:
    """Retrieve ``labeled`` with ``retriever`` and score it against the answer spans.

    ``corpus_chunks`` is the full ingested corpus; its (ticker-scoped) relevant count is the
    recall denominator. Retrieval is scoped by ``labeled.ticker`` (same filter used in prod).
    """
    k = labeled.top_k or top_k
    where = ChunkFilter(ticker=labeled.ticker) if labeled.ticker else None
    evidence = retriever.retrieve(labeled.query, top_k=k, where=where)
    flags = [labeled.is_relevant(rc.chunk) for rc in evidence.chunks]
    n_relevant = sum(
        1
        for c in corpus_chunks
        if (labeled.ticker is None or c.ticker == labeled.ticker) and labeled.is_relevant(c)
    )
    return QueryReport(
        query=labeled.query,
        k=k,
        n_relevant_corpus=n_relevant,
        n_retrieved=len(flags),
        hit=hit_at_k(flags, k),
        reciprocal_rank=reciprocal_rank(flags, k),
        precision=precision_at_k(flags, k),
        recall=recall_at_k(flags, n_relevant, k),
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
        retriever = Retriever(embedder, store)
        per_query = [
            evaluate_query(retriever, q, top_k=top_k, corpus_chunks=corpus_chunks) for q in queries
        ]
        reports.append(
            EmbedderReport(
                embedder=name,
                top_k=top_k,
                n_queries=len(per_query),
                hit_at_k=_mean([q.hit for q in per_query]),
                mrr=_mean([q.reciprocal_rank for q in per_query]),
                precision_at_k=_mean([q.precision for q in per_query]),
                recall_at_k=_mean([q.recall for q in per_query if q.n_relevant_corpus > 0]),
                per_query=per_query,
            )
        )
    return reports


def format_reports_markdown(reports: Sequence[EmbedderReport]) -> str:
    """Render the A/B reports as a GitHub-flavored Markdown comparison table (CLI output)."""
    if not reports:
        return "_no embedders evaluated_"
    k = reports[0].top_k
    lines = [
        f"| embedder | hit@{k} | MRR | precision@{k} | recall@{k} | queries |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.embedder} | {r.hit_at_k:.3f} | {r.mrr:.3f} | "
            f"{r.precision_at_k:.3f} | {r.recall_at_k:.3f} | {r.n_queries} |"
        )
    return "\n".join(lines)
