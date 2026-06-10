"""Embedding backends behind one ``Embedder`` Protocol (RAG P4).

Turns chunk/query text into dense vectors. Local ``fastembed``/BGE is the default
(onnxruntime, no torch, $0); ``OpenAIEmbedder`` is the opt-in alternative behind the
same Protocol, so swapping providers never touches the chunking, store, or retrieval
code. ``build_embedder(settings)`` selects by config.

Cost/latency discipline (see rag_concepts.md §3.1): documents are embedded ONCE at
ingestion; only a single query string is embedded per search. Heavy backends are
**lazy-imported** and models load on first ``embed``, so importing this module and
constructing an embedder are cheap + offline — CI never downloads a model (tests use
``FakeEmbedder``).

Install the real backends with the extras:  ``pip install -e ".[rag]"`` (fastembed),
``pip install -e ".[openai]"`` (OpenAI).
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from stock_agent.settings import Settings

# Output dimensionality of the OpenAI embedding models (no model load needed to know it).
_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_OPENAI_DEFAULT_MODEL = "text-embedding-3-small"

# Voyage models output 1024-d by default. Current lineup (the voyage-4 family is
# newest, with a 200M-token free allotment; the domain "-2" models — finance/law/code
# — have 50M free). Default = voyage-4 (newest, cheap, 200M free covers our corpus);
# voyage-finance-2 is the finance-tuned alternative to A/B against after P8.
_VOYAGE_DIMS = {
    "voyage-4-large": 1024,
    "voyage-4": 1024,
    "voyage-4-lite": 1024,
    "voyage-context-3": 1024,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
    "voyage-code-2": 1024,
}
_VOYAGE_DEFAULT_MODEL = "voyage-4"


@runtime_checkable
class Embedder(Protocol):
    """A text -> vector encoder. Same model must embed documents AND queries (§4.2)."""

    name: str

    @property
    def dim(self) -> int:
        """Embedding dimensionality (vectors are length-``dim``)."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus passages (used once at ingestion)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query (may use a model-specific query prefix)."""
        ...


def _l2_normalize(vec: list[float]) -> list[float]:
    """Scale to unit length so dot product == cosine similarity (§4.2)."""
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeEmbedder:
    """Deterministic, dependency-free embedder for tests (and downstream P5/P6).

    Hashes text to a fixed-dim unit vector — same text -> same vector, different text
    -> (almost surely) different vector — so retrieval logic is testable without any
    model or network. NOT semantic; for tests only.
    """

    name = "fake"

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        # Stretch a SHA-256 digest into `dim` floats in [-1, 1], then unit-normalize.
        raw = b""
        counter = 0
        while len(raw) < self._dim * 4:
            raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
            counter += 1
        floats = [struct.unpack("<I", raw[i * 4 : i * 4 + 4])[0] for i in range(self._dim)]
        return _l2_normalize([(f / 0xFFFFFFFF) * 2.0 - 1.0 for f in floats])

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FastEmbedEmbedder:
    """Local embeddings via ``fastembed`` (default: ``BAAI/bge-small-en-v1.5``, 384-d).

    onnxruntime under the hood (no torch -> avoids the macOS torch+lightgbm OpenMP
    segfault). The model is loaded lazily on first use, so construction is free.
    """

    name = "fastembed"

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._dim: int | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "Local embeddings need fastembed: pip install -e \".[rag]\""
                ) from exc
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._ensure_model()
        return [[float(x) for x in vec] for vec in model.embed(list(texts))]  # type: ignore[attr-defined]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        # Retrieval models (BGE/E5) add a query-instruction prefix via query_embed;
        # fall back to plain embed on backends that don't expose it.
        query_embed = getattr(model, "query_embed", None)
        gen = query_embed([text]) if callable(query_embed) else model.embed([text])  # type: ignore[attr-defined]
        return [float(x) for x in next(iter(gen))]


class OpenAIEmbedder:
    """OpenAI embeddings (opt-in). Default model ``text-embedding-3-small`` (1536-d)."""

    name = "openai"

    def __init__(
        self,
        settings: Settings,
        *,
        model: str = _OPENAI_DEFAULT_MODEL,
        client: object | None = None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._client = client  # injected in tests

    def _ensure_client(self) -> object:
        if self._client is None:
            # Check the key BEFORE importing, so a missing key gives the clear settings error.
            key = self._settings.require("openai_api_key", capability="OpenAI embeddings")
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "OpenAI embeddings need the openai package: pip install -e \".[openai]\""
                ) from exc
            self._client = openai.OpenAI(api_key=key)
        return self._client

    @property
    def dim(self) -> int:
        return _OPENAI_DIMS.get(self._model, 1536)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self._ensure_client().embeddings.create(  # type: ignore[attr-defined]
            model=self._model, input=list(texts)
        )
        return [[float(x) for x in item.embedding] for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class VoyageEmbedder:
    """Voyage AI embeddings (opt-in). Default ``voyage-4`` (1024-d).

    Voyage is Anthropic's recommended embedding provider. ``voyage-4`` is the newest
    general model (200M-token free pool, then $0.06/1M); pass ``model="voyage-finance-2"``
    to A/B the finance-tuned variant. Uses Voyage's ``input_type`` asymmetry —
    ``"document"`` for passages, ``"query"`` for searches — which lifts retrieval
    quality. Charges a ``VOYAGE_API_KEY`` (independent of Anthropic/OpenAI).
    """

    name = "voyage"

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
            key = self._settings.require("voyage_api_key", capability="Voyage embeddings")
            try:
                import voyageai
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "Voyage embeddings need the voyageai package: pip install -e \".[voyage]\""
                ) from exc
            self._client = voyageai.Client(api_key=key)
        return self._client

    @property
    def dim(self) -> int:
        return _VOYAGE_DIMS.get(self._model, 1024)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        result = self._ensure_client().embed(  # type: ignore[attr-defined]
            list(texts), model=self._model, input_type="document"
        )
        return [[float(x) for x in vec] for vec in result.embeddings]

    def embed_query(self, text: str) -> list[float]:
        result = self._ensure_client().embed(  # type: ignore[attr-defined]
            [text], model=self._model, input_type="query"
        )
        return [float(x) for x in result.embeddings[0]]


def build_embedder(settings: Settings) -> Embedder:
    """Select the configured embedder (``settings.embedding_provider``).

    ``local`` (default) -> ``FastEmbedEmbedder(settings.embedding_model)``;
    ``openai`` -> ``OpenAIEmbedder`` (text-embedding-3-small); ``voyage`` ->
    ``VoyageEmbedder`` (voyage-4). None loads a model or a client here (all lazy),
    so this is cheap and offline.
    """
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings)
    if settings.embedding_provider == "voyage":
        return VoyageEmbedder(settings)
    return FastEmbedEmbedder(settings.embedding_model)
