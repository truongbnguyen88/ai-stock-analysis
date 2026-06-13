"""Reranking (advanced-RAG A2) — retrieve wide, then narrow with a cross-encoder.

Dense retrieval scores a query against pre-computed passage vectors with a **bi-encoder**: the
query and each chunk are embedded *independently*, so the model never sees them together and the
score is a coarse cosine. A **cross-encoder** instead feeds ``(query, chunk)`` through the model
*jointly* and emits a direct relevance score — far more accurate, but too expensive to run over the
whole corpus. The standard fix is two-stage: cheap bi-encoder retrieves a wide candidate set
(``fetch_k`` ≈ 30), the cross-encoder rescores just those and we keep the top ``top_k`` (5–8).

Everything here is **default-OFF**: with ``rerank_provider="none"`` the factory returns the plain
``Retriever`` and the pipeline is byte-identical to A1. Rerankers sit behind a ``Reranker`` Protocol
(local ``fastembed`` onnx cross-encoder — **no torch**; Voyage ``rerank`` API; or a no-op), and
``RerankingRetriever`` wraps ANY ``RetrievalSystem`` (dense today, hybrid at A3) so the eval
harness, ``research/``, and the agent tools never change. Reranking only **reorders** chunks — it
never edits their text — so citations and number-grounding are untouched.

Install the local reranker with the RAG extra: ``pip install -e ".[rag]"`` (ships the encoder).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from stock_agent.rag.embeddings import build_embedder
from stock_agent.rag.retriever import RetrievalSystem, Retriever
from stock_agent.rag.vector_store import VectorStore, build_vector_store
from stock_agent.schemas.retrieval import ChunkFilter, EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings

# Provider defaults (used when ``settings.rerank_model`` is unset). The local default is a small,
# fast onnx cross-encoder; the Voyage default is its current general reranker.
_LOCAL_DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_VOYAGE_DEFAULT_MODEL = "rerank-2"


@runtime_checkable
class Reranker(Protocol):
    """Rescore ``(query, chunk)`` pairs and return the chunks reordered (best first).

    Implementations return a NEW list with rerank scores in ``RetrievedChunk.score``; they never
    mutate chunk text. ``name`` labels the reranker in eval reports.
    """

    name: str

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return ``chunks`` reordered by joint relevance to ``query`` (descending)."""
        ...


class NoOpReranker:
    """Identity reranker — returns the input order unchanged. The default (rerank disabled)."""

    name = "noop"

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return list(chunks)


class FastEmbedReranker:
    """Local cross-encoder via ``fastembed`` (default ``Xenova/ms-marco-MiniLM-L-6-v2``).

    onnxruntime under the hood (**no torch** — avoids the macOS torch+lightgbm OpenMP segfault that
    drove the embedder choice). The model loads lazily on first ``rerank``, so construction is free
    and offline; CI never downloads it (tests use a ``FakeReranker``).
    """

    name = "fastembed-rerank"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    'Local reranking needs fastembed: pip install -e ".[rag]"'
                ) from exc
            self._model = TextCrossEncoder(model_name=self._model_name)
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self._ensure_model()
        # TextCrossEncoder.rerank(query, docs) -> one relevance score per doc, input order.
        scores = list(model.rerank(query, [rc.chunk.text for rc in chunks]))  # type: ignore[attr-defined]
        ranked = sorted(zip(scores, chunks, strict=True), key=lambda t: t[0], reverse=True)
        return [RetrievedChunk(chunk=rc.chunk, score=float(score)) for score, rc in ranked]


class VoyageReranker:
    """Voyage ``rerank`` API (opt-in; reuses ``VOYAGE_API_KEY``). Default model ``rerank-2``.

    No local model — Voyage scores the ``(query, chunk)`` pairs server-side and returns them sorted.
    Charges tokens, so it is opt-in (``rerank_provider="voyage"``); the client is lazy + injectable
    for tests.
    """

    name = "voyage-rerank"

    def __init__(
        self,
        settings: Settings,
        *,
        model: str = _VOYAGE_DEFAULT_MODEL,
        client: object | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._client = client  # injected in tests

    def _ensure_client(self) -> object:
        if self._client is None:
            key = self._settings.require("voyage_api_key", capability="Voyage reranking")
            try:
                import voyageai
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    'Voyage reranking needs the voyageai package: pip install -e ".[voyage]"'
                ) from exc
            self._client = voyageai.Client(api_key=key)
        return self._client

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        result = self._ensure_client().rerank(  # type: ignore[attr-defined]
            query, [rc.chunk.text for rc in chunks], model=self._model
        )
        # result.results: each has .index (into the input) and .relevance_score, sorted best-first.
        return [
            RetrievedChunk(chunk=chunks[r.index].chunk, score=float(r.relevance_score))
            for r in result.results
        ]


def build_reranker(settings: Settings) -> Reranker:
    """Select the configured reranker (``settings.rerank_provider``); ``rerank_model`` overrides
    the per-provider default. Nothing loads a model or client here (all lazy), so this is cheap."""
    provider = settings.rerank_provider
    if provider == "local":
        return FastEmbedReranker(settings.rerank_model or _LOCAL_DEFAULT_MODEL)
    if provider == "voyage":
        return VoyageReranker(settings, model=settings.rerank_model or _VOYAGE_DEFAULT_MODEL)
    return NoOpReranker()


class RerankingRetriever:
    """Wrap a base ``RetrievalSystem`` with a two-stage retrieve-wide → rerank → narrow.

    Over-fetches ``fetch_k`` candidates from ``base`` (at least ``top_k``), rescores them with the
    cross-encoder, and returns the top ``top_k``. Satisfies ``RetrievalSystem``, so it composes:
    today it wraps the dense ``Retriever``; at A3 it can wrap the hybrid retriever unchanged.
    """

    def __init__(self, base: RetrievalSystem, reranker: Reranker, *, fetch_k: int = 30) -> None:
        self._base = base
        self._reranker = reranker
        self._fetch_k = fetch_k

    @property
    def name(self) -> str:
        return f"rerank({self._reranker.name})+{self._base.name}"

    def retrieve(self, query: str, *, top_k: int, where: ChunkFilter | None = None) -> EvidenceSet:
        # Fetch a wide candidate set (never fewer than top_k), rerank, then keep top_k.
        wide = self._base.retrieve(query, top_k=max(self._fetch_k, top_k), where=where)
        if wide.is_empty:
            return wide  # nothing to rerank → preserve the empty-retrieval refusal path
        reranked = self._reranker.rerank(query, list(wide.chunks))
        return EvidenceSet(query=query, chunks=reranked[:top_k])


def build_retrieval_system(
    settings: Settings, *, store: VectorStore | None = None
) -> RetrievalSystem:
    """Build the configured read-path retriever — the single gated factory the app wires in.

    Returns the dense ``Retriever`` by default; wraps it in a ``RerankingRetriever`` only when
    ``rerank_provider != "none"``. ``store`` is injectable (tests pass an in-memory store; prod lets
    it default to ``build_vector_store``). All backends are lazy, so this stays cheap and offline.
    """
    base = Retriever(build_embedder(settings), store or build_vector_store(settings))
    if settings.rerank_provider == "none":
        return base
    return RerankingRetriever(base, build_reranker(settings), fetch_k=settings.rerank_fetch_k)
