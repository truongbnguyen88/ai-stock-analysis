"""Embedding backends (RAG P4) — FakeEmbedder, selector, OpenAI (fake client).

The real fastembed model is exercised only by a gated test (RUN_EMBED_TESTS) so CI
never downloads a model; everything else is offline + deterministic.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pytest

from stock_agent.rag.embeddings import (
    Embedder,
    FakeEmbedder,
    FastEmbedEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    build_embedder,
    embedding_namespace,
)
from stock_agent.settings import MissingSettingError, Settings


# ---- FakeEmbedder ------------------------------------------------------------
def test_fake_embedder_deterministic() -> None:
    e = FakeEmbedder(dim=16)
    assert e.embed_query("risk factors") == e.embed_query("risk factors")
    assert e.embed_query("risk factors") != e.embed_query("md&a")


def test_fake_embedder_shape_and_unit_norm() -> None:
    e = FakeEmbedder(dim=16)
    vecs = e.embed_documents(["a", "bb", "ccc"])
    assert len(vecs) == 3 and all(len(v) == 16 for v in vecs)
    for v in vecs:
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)


def test_fake_embedder_satisfies_protocol() -> None:
    assert isinstance(FakeEmbedder(), Embedder)


# ---- build_embedder selector (no model load / no network) --------------------
def test_build_embedder_local_default() -> None:
    e = build_embedder(Settings(_env_file=None))
    assert isinstance(e, FastEmbedEmbedder)
    assert e.name == "fastembed"
    assert e._model_name == "BAAI/bge-small-en-v1.5"  # from settings.embedding_model


def test_build_embedder_openai() -> None:
    e = build_embedder(Settings(_env_file=None, embedding_provider="openai"))
    assert isinstance(e, OpenAIEmbedder)
    assert e.name == "openai"


def test_build_embedder_voyage() -> None:
    e = build_embedder(Settings(_env_file=None, embedding_provider="voyage"))
    assert isinstance(e, VoyageEmbedder)
    assert e.name == "voyage"


# ---- OpenAIEmbedder via an injected fake client ------------------------------
class _FakeEmbItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [_FakeEmbItem(e) for e in embeddings]


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.last_call: tuple[str, list[str]] | None = None

    def create(self, *, model: str, input: list[str]) -> _FakeEmbResponse:
        self.last_call = (model, list(input))
        return _FakeEmbResponse([[0.1, 0.2, 0.3] for _ in input])


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


def test_openai_embedder_with_fake_client() -> None:
    client = _FakeOpenAIClient()
    e = OpenAIEmbedder(Settings(_env_file=None), client=client)
    vecs = e.embed_documents(["chip revenue", "gaming"])
    assert vecs == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert client.embeddings.last_call == ("text-embedding-3-small", ["chip revenue", "gaming"])
    assert e.embed_query("query") == [0.1, 0.2, 0.3]
    assert e.dim == 1536  # known dim, no model load


def test_openai_embedder_requires_key() -> None:
    # No injected client + no key -> the settings error fires before any import.
    e = OpenAIEmbedder(Settings(_env_file=None))
    with pytest.raises(MissingSettingError):
        e.embed_documents(["x"])


# ---- VoyageEmbedder via an injected fake client ------------------------------
class _FakeVoyageResult:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class _FakeVoyageClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    def embed(self, texts: list[str], *, model: str, input_type: str) -> _FakeVoyageResult:
        self.calls.append((model, input_type, list(texts)))
        return _FakeVoyageResult([[0.5, 0.6] for _ in texts])


def test_voyage_embedder_uses_input_type_and_default_model() -> None:
    client = _FakeVoyageClient()
    e = VoyageEmbedder(Settings(_env_file=None), client=client)
    assert e.embed_documents(["risk factors"]) == [[0.5, 0.6]]
    assert e.embed_query("query") == [0.5, 0.6]
    # Voyage's asymmetric input_type: "document" for passages, "query" for searches.
    assert client.calls == [
        ("voyage-4", "document", ["risk factors"]),
        ("voyage-4", "query", ["query"]),
    ]
    assert e.dim == 1024  # known dim, no model load


def test_voyage_embedder_requires_key() -> None:
    e = VoyageEmbedder(Settings(_env_file=None))
    with pytest.raises(MissingSettingError):
        e.embed_documents(["x"])


# ---- real fastembed model (gated; never runs in CI) --------------------------
@pytest.mark.skipif(
    not os.environ.get("RUN_EMBED_TESTS"),
    reason="downloads the BGE model; run locally with RUN_EMBED_TESTS=1 and the [rag] extra",
)
def test_fastembed_real_model() -> None:
    pytest.importorskip("fastembed")
    e: Any = FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    assert e.dim == 384
    assert e.embed_query("AI data-center demand") == e.embed_query("AI data-center demand")
    docs = e.embed_documents(["chip revenue grew", "gaming declined"])
    assert len(docs) == 2 and all(len(v) == 384 for v in docs)


# ---- 9c: embedder namespace (store isolation across providers) ---------------


def test_embedding_namespace_distinguishes_providers() -> None:
    local = embedding_namespace(Settings(_env_file=None, embedding_provider="local"))
    voyage = embedding_namespace(Settings(_env_file=None, embedding_provider="voyage"))
    openai = embedding_namespace(Settings(_env_file=None, embedding_provider="openai"))
    assert local != voyage != openai and local != openai
    assert "voyage-4" in voyage  # default voyage model is reflected
    # The local namespace carries the model, so two BGE variants don't collide either.
    other = embedding_namespace(
        Settings(
            _env_file=None, embedding_provider="local", embedding_model="BAAI/bge-base-en-v1.5"
        )
    )
    assert other != local
