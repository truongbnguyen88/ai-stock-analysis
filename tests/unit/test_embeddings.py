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


def test_voyage_embedder_batches_large_inputs(monkeypatch: Any) -> None:
    # A corpus embed is thousands of chunks; the provider caps each request, so embed_documents
    # must split into batches. With the item cap forced to 2, 5 texts -> 3 requests (2+2+1).
    import stock_agent.rag.embeddings as emb

    monkeypatch.setattr(emb, "_EMBED_BATCH_MAX_ITEMS", 2)
    client = _FakeVoyageClient()
    e = VoyageEmbedder(Settings(_env_file=None), client=client)
    texts = [f"document number {i}" for i in range(5)]
    vecs = e.embed_documents(texts)
    assert len(vecs) == 5  # one vector per input, order preserved
    assert [len(call[2]) for call in client.calls] == [2, 2, 1]  # batched requests
    assert all(call[1] == "document" for call in client.calls)


def test_voyage_embed_documents_empty_makes_no_call() -> None:
    client = _FakeVoyageClient()
    e = VoyageEmbedder(Settings(_env_file=None), client=client)
    assert e.embed_documents([]) == []
    assert client.calls == []  # no empty request sent


# ---- Voyage transient-retry wrapper (e5_v2 resilience fix) -------------------
# Regression guard: long RL retrains issue thousands of live embeds and died on transient
# `APIConnectionError`, which the voyageai SDK's own controller does NOT retry. These test the
# app-level wrapper deterministically — monkeypatched transient set + no-op sleep, so they run
# with or without the (CI-absent) voyage extra and never actually sleep.
class _Transient(Exception):
    """Stand-in for a Voyage transient error, independent of the optional voyageai package."""


def _no_backoff(monkeypatch: Any) -> None:
    import stock_agent.rag.embeddings as emb

    # Zero the backoff base so retries don't actually sleep (delay = base * 2**attempt = 0).
    monkeypatch.setattr(emb, "_VOYAGE_RETRY_BASE_S", 0.0)


def test_voyage_transient_excs_returns_tuple() -> None:
    # Locally (voyage extra present) it names APIConnectionError — the exact error the SDK
    # skips; in CI (extra absent) the ImportError branch degrades to () — see the helper.
    import stock_agent.rag.embeddings as emb

    excs = emb._voyage_transient_excs()
    assert isinstance(excs, tuple)
    from voyageai import error as ve  # dev env has the extra; asserts the set is the intended one

    assert ve.APIConnectionError in excs


def test_voyage_retry_retries_transient_then_succeeds(monkeypatch: Any) -> None:
    import stock_agent.rag.embeddings as emb

    monkeypatch.setattr(emb, "_voyage_transient_excs", lambda: (_Transient,))
    _no_backoff(monkeypatch)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:  # fail twice, then succeed on the 3rd attempt
            raise _Transient("connection aborted")
        return "ok"

    assert emb._voyage_call_with_retry(flaky) == "ok"
    assert calls["n"] == 3  # retried, did not give up early


def test_voyage_retry_exhausts_after_max_attempts(monkeypatch: Any) -> None:
    import stock_agent.rag.embeddings as emb

    monkeypatch.setattr(emb, "_voyage_transient_excs", lambda: (_Transient,))
    _no_backoff(monkeypatch)
    calls = {"n": 0}

    def always_fails() -> str:
        calls["n"] += 1
        raise _Transient("still down")

    with pytest.raises(_Transient):
        emb._voyage_call_with_retry(always_fails)
    assert calls["n"] == emb._VOYAGE_APP_RETRIES  # bounded attempts, then the last error surfaces


def test_voyage_retry_propagates_non_transient_immediately(monkeypatch: Any) -> None:
    # An auth / malformed-request error is not in the transient set -> raise on the first try.
    import stock_agent.rag.embeddings as emb

    monkeypatch.setattr(emb, "_voyage_transient_excs", lambda: (_Transient,))
    _no_backoff(monkeypatch)
    calls = {"n": 0}

    def bad_request() -> str:
        calls["n"] += 1
        raise ValueError("401 unauthorized")

    with pytest.raises(ValueError):
        emb._voyage_call_with_retry(bad_request)
    assert calls["n"] == 1  # no retry on a non-transient error


def test_voyage_embed_works_when_transient_set_empty(monkeypatch: Any) -> None:
    # Simulates the CI path: voyageai absent -> _voyage_transient_excs() == () -> the wrapper
    # must still issue the (fake-client) embed exactly once and return its vectors unchanged.
    import stock_agent.rag.embeddings as emb

    monkeypatch.setattr(emb, "_voyage_transient_excs", lambda: ())
    client = _FakeVoyageClient()
    e = VoyageEmbedder(Settings(_env_file=None), client=client)
    assert e.embed_documents(["risk factors"]) == [[0.5, 0.6]]
    assert e.embed_query("query") == [0.5, 0.6]


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
