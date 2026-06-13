"""Sparse BM25 store (advanced-RAG A3) — FTS5 + in-memory backends, behaviour parity.

The same behavioural tests run against BOTH backends (parametrized), so the pure-Python BM25 and
the SQLite FTS5 index stay interchangeable. FTS5 uses a temp DB; no network, no model.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from stock_agent.rag.sparse_store import (
    Fts5SparseStore,
    InMemoryBM25Store,
    SparseStore,
    build_sparse_store,
    fts5_available,
)
from stock_agent.schemas.documents import DocumentChunk
from stock_agent.schemas.retrieval import ChunkFilter
from stock_agent.settings import Settings


def _chunk(
    idx: int, text: str, *, ticker: str = "NVDA", section: str = "Item 1A. Risk Factors"
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{ticker}:10-K:2026-02-25:{idx}",
        document_id=f"{ticker}:10-K:2026-02-25",
        chunk_index=idx,
        text=text,
        ticker=ticker,
        document_type="10-K",
        source="SEC",
        source_url=f"https://www.sec.gov/Archives/edgar/data/x/{ticker.lower()}.htm",
        filing_date=date(2026, 2, 25),
        section=section,
    )


@pytest.fixture(params=["fts5", "mem"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> SparseStore:
    if request.param == "fts5":
        if not fts5_available():  # pragma: no cover - environment-dependent
            pytest.skip("sqlite3 built without FTS5")
        return Fts5SparseStore(tmp_path / "sparse.db")
    return InMemoryBM25Store()


_C0 = _chunk(0, "Export controls restrict China sales of advanced chips.")
_C1 = _chunk(1, "Export licensing requirements apply to certain products.")
_C2 = _chunk(2, "Data center revenue grew on strong demand.")


def test_search_ranks_term_matches_and_excludes_non_matches(store: SparseStore) -> None:
    store.add([_C0, _C1, _C2])
    hits = store.search("export china", top_k=10)
    ids = [cid for cid, _ in hits]
    assert _C2.chunk_id not in ids  # no query term → not returned
    assert ids[0] == _C0.chunk_id  # contains BOTH query terms → outranks the one-term chunk
    assert _C1.chunk_id in ids
    assert all(s0 >= s1 for (_, s0), (_, s1) in zip(hits, hits[1:], strict=False))  # desc scores


def test_search_empty_or_no_match_returns_empty(store: SparseStore) -> None:
    store.add([_C0])
    assert store.search("", top_k=5) == []
    assert store.search("nonexistentterm", top_k=5) == []
    assert store.search("export", top_k=0) == []


def test_metadata_filter_scopes_results(store: SparseStore) -> None:
    amd = _chunk(0, "Export controls also affect our China business.", ticker="AMD")
    store.add([_C0, _C1, amd])
    hits = store.search("export", top_k=10, where=ChunkFilter(ticker="NVDA"))
    ids = {cid for cid, _ in hits}
    assert ids == {_C0.chunk_id, _C1.chunk_id}  # AMD excluded by the filter


def test_existing_ids_and_fetch(store: SparseStore) -> None:
    store.add([_C0, _C1])
    assert store.existing_ids([_C0.chunk_id, "missing"]) == {_C0.chunk_id}
    fetched = store.fetch([_C0.chunk_id, "missing"])
    assert set(fetched) == {_C0.chunk_id}
    assert fetched[_C0.chunk_id].text == _C0.text  # full chunk materialized (text + metadata)
    assert fetched[_C0.chunk_id].ticker == "NVDA"


def test_add_is_idempotent_upsert(store: SparseStore) -> None:
    store.add([_C0, _C1])
    store.add([_C0, _C1])  # re-add same ids
    assert store.count() == 2  # upsert by chunk_id, not duplicated


def test_section_and_date_filters(store: SparseStore) -> None:
    mdna = _chunk(3, "Export trends discussed by management.", section="Item 7. MD&A")
    store.add([_C0, mdna])
    risk_only = store.search(
        "export", top_k=10, where=ChunkFilter(sections=["Item 1A. Risk Factors"])
    )
    assert {cid for cid, _ in risk_only} == {_C0.chunk_id}
    in_window = store.search(
        "export", top_k=10, where=ChunkFilter(date_from=date(2026, 1, 1), date_to=date(2026, 3, 1))
    )
    assert _C0.chunk_id in {cid for cid, _ in in_window}
    none_window = store.search(
        "export", top_k=10, where=ChunkFilter(date_from=date(2027, 1, 1))
    )
    assert none_window == []


def test_protocol_conformance(store: SparseStore) -> None:
    assert isinstance(store, SparseStore)


def test_build_sparse_store_selects_a_backend(tmp_path: Path) -> None:
    s = build_sparse_store(Settings(_env_file=None, sparse_store_dir=tmp_path))
    assert isinstance(s, SparseStore)  # FTS5 where available, else in-memory
