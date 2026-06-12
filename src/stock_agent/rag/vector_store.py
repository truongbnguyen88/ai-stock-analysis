"""Vector store (RAG P5) — persist chunk embeddings + metadata-filtered top-k search.

Stores **pre-computed** vectors (the ``Embedder`` runs in P6's pipeline, not here) keyed by
``chunk_id``, and serves cosine-similarity nearest-neighbour search scoped by a ``ChunkFilter``
(ticker / form / section / date). NO LLM and NO embedding happen here — this is a pure
storage + ANN layer, embedder-agnostic.

Two backends behind one ``VectorStore`` Protocol:
- ``InMemoryVectorStore`` — pure-Python cosine + filter, no persistence. Deterministic and
  dependency-free, so it runs in CI and doubles as a chromadb-free fallback / P6 test backend.
- ``ChromaVectorStore`` — persistent (local Chroma), the default for the real app.

Both expose **similarity** scores (higher = closer). The Chroma collection uses cosine space,
so Chroma returns a cosine *distance*; we convert to similarity as ``score = 1 - distance``
(our embedders emit unit-norm vectors, so this is exactly cosine similarity — matching
``RetrievedChunk.score``). See docs/rag_concepts.md §4.2.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import date as Date
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from stock_agent.rag.embeddings import embedding_namespace
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter, RetrievedChunk
from stock_agent.settings import Settings

_COLLECTION = "filings"  # base name; the persistent collection is namespaced per embedder


@runtime_checkable
class VectorStore(Protocol):
    """Persist chunk vectors + metadata; serve cosine top-k with metadata filtering."""

    name: str

    def add(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> None:
        """Upsert ``chunks`` with their pre-computed ``vectors`` (1:1, keyed by ``chunk_id``)."""
        ...

    def query(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        where: ChunkFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` chunks most similar to ``query_vector`` (filtered, ranked)."""
        ...

    def count(self) -> int:
        """Number of stored chunks."""
        ...

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Subset of ``ids`` already stored — lets incremental ingest skip re-embedding them."""
        ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity (robust to non-unit vectors; unit vectors -> plain dot product)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class InMemoryVectorStore:
    """Pure-Python store: cosine top-k + metadata filter, no persistence.

    Holds chunks/vectors in dicts keyed by ``chunk_id`` (re-adding an id upserts). Used in
    CI (chromadb is an optional extra) and usable standalone for small/offline corpora.
    """

    name = "memory"

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def add(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = list(vector)

    def query(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        where: ChunkFilter | None = None,
    ) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=chunk, score=_cosine(query_vector, self._vectors[cid]))
            for cid, chunk in self._chunks.items()
            if where is None or where.matches(chunk)
        ]
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        return {cid for cid in ids if cid in self._chunks}


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    """Flat, Chroma-storable metadata (scalars only; ``section`` omitted when None).

    ``filing_date`` is stored twice: ISO string for citation/equality, and an integer
    ``filing_date_ord`` (YYYYMMDD) so date ranges can use Chroma's numeric ``$gte``/``$lte``.
    """
    md: dict[str, Any] = {
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "ticker": chunk.ticker,
        "document_type": chunk.document_type,
        "source": chunk.source,
        "source_url": chunk.source_url,
        "filing_date": chunk.filing_date.isoformat(),
        "filing_date_ord": int(chunk.filing_date.strftime("%Y%m%d")),
    }
    if chunk.section is not None:
        md["section"] = chunk.section  # Chroma rejects None — omit instead
    return md


def _chunk_from(chunk_id: str, text: str, md: Mapping[str, Any]) -> DocumentChunk:
    """Rebuild a ``DocumentChunk`` from a Chroma id + document text + metadata."""
    section = md.get("section")
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=str(md["document_id"]),
        chunk_index=int(md["chunk_index"]),
        text=text,
        ticker=str(md["ticker"]),
        document_type=str(md["document_type"]),  # validated against the Literal by pydantic
        source=str(md["source"]),
        source_url=str(md["source_url"]),
        filing_date=Date.fromisoformat(str(md["filing_date"])),
        section=str(section) if section is not None else None,
    )


def _to_chroma_where(f: ChunkFilter) -> dict[str, Any] | None:
    """Translate a ``ChunkFilter`` into a Chroma ``where`` clause (``$and`` of conditions)."""
    conds: list[dict[str, Any]] = []
    if f.ticker is not None:
        conds.append({"ticker": f.ticker})
    if f.document_types is not None:
        conds.append({"document_type": {"$in": list(f.document_types)}})
    if f.sections is not None:
        conds.append({"section": {"$in": list(f.sections)}})
    if f.date_from is not None:
        conds.append({"filing_date_ord": {"$gte": int(f.date_from.strftime("%Y%m%d"))}})
    if f.date_to is not None:
        conds.append({"filing_date_ord": {"$lte": int(f.date_to.strftime("%Y%m%d"))}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


class ChromaVectorStore:
    """Persistent local vector store backed by Chroma (cosine space).

    ``chromadb`` is imported lazily (it is the ``[rag]`` extra, absent in CI). The collection
    is created with ``hnsw:space="cosine"``; queries convert cosine distance to similarity.
    """

    name = "chroma"

    def __init__(
        self,
        path: Path,
        *,
        collection_name: str = _COLLECTION,
        client: Any | None = None,
    ) -> None:
        self._path = path
        self._collection_name = collection_name
        self._client: Any = client  # injected in tests
        self._collection: Any = None

    def _ensure(self) -> Any:
        if self._collection is None:
            if self._client is None:
                try:
                    import chromadb
                    from chromadb.config import Settings as ChromaSettings
                except ImportError as exc:  # pragma: no cover - only without the extra
                    raise RuntimeError(
                        'Chroma vector store needs the chromadb package: pip install -e ".[rag]"'
                    ) from exc
                self._path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self._path),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(self, chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> None:
        items = list(chunks)
        if not items:
            return  # Chroma rejects an empty upsert
        self._ensure().upsert(
            ids=[c.chunk_id for c in items],
            embeddings=[list(v) for v in vectors],
            documents=[c.text for c in items],
            metadatas=[_chunk_metadata(c) for c in items],
        )

    def query(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        where: ChunkFilter | None = None,
    ) -> list[RetrievedChunk]:
        clause = _to_chroma_where(where) if where is not None and not where.is_empty else None
        res = self._ensure().query(
            query_embeddings=[list(query_vector)],
            n_results=top_k,
            where=clause,
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            RetrievedChunk(chunk=_chunk_from(cid, text, md), score=1.0 - float(dist))
            for cid, text, md, dist in zip(ids, docs, metas, dists, strict=True)
        ]

    def count(self) -> int:
        return int(self._ensure().count())

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        if not ids:
            return set()
        # Chroma's get(ids=...) returns only the ids that exist (missing ones are silently absent);
        # include=[] skips fetching documents/metadata/embeddings — we only need the id set.
        res = self._ensure().get(ids=list(ids), include=[])
        return set(res["ids"])


def collection_name_for(namespace: str) -> str:
    """Chroma-safe collection name for an embedder ``namespace`` (per-embedder isolation).

    Chroma requires 3–512 chars of ``[a-zA-Z0-9._-]`` starting/ending alphanumeric. We slugify
    the namespace, e.g. ``local-BAAI/bge-small-en-v1.5`` -> ``filings-local-baai-bge-small-en-v1-5``
    and ``voyage-voyage-4`` -> ``filings-voyage-voyage-4``. The ``filings-`` prefix guarantees a
    valid leading char; the lowercased alnum-only slug guarantees a valid trailing char.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", namespace).strip("-").lower()
    return f"{_COLLECTION}-{slug}" if slug else _COLLECTION


def build_vector_store(settings: Settings) -> VectorStore:
    """Construct the configured persistent store (Chroma at ``settings.vector_store_dir``).

    The collection is **namespaced per embedder** (``embedding_namespace``): corpora embedded
    with different models have different vector dimensions and must not share a collection.
    Switching ``embedding_provider`` (e.g. ``local`` -> ``voyage``, RAG_TODO 9c) therefore
    targets a fresh collection; a re-ingest populates it without colliding with the old one.
    """
    return ChromaVectorStore(
        settings.vector_store_dir,
        collection_name=collection_name_for(embedding_namespace(settings)),
    )
