"""RAG ingest pipeline (P6) — disk filings → chunks → store, with FakeEmbedder (offline)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from stock_agent.documents.parsers import load_filing
from stock_agent.rag.chunking import chunk_filing, estimate_tokens
from stock_agent.rag.embeddings import FakeEmbedder
from stock_agent.rag.pipeline import (
    BulkIngestResult,
    EmbedBudgetExceeded,
    IngestResult,
    backfill_sparse,
    bulk_ingest,
    ingest_ticker,
    iter_filing_dirs,
)
from stock_agent.rag.retriever import Retriever
from stock_agent.rag.sparse_store import InMemoryBM25Store
from stock_agent.rag.vector_store import InMemoryVectorStore
from stock_agent.schemas.retrieval import ChunkFilter

_EMB = FakeEmbedder(dim=32)

_HTML = (
    "<html><body>"
    "<div>Item 1A. Risk Factors</div>"
    "<p>Our supply chain and export-control exposure could materially harm our results. "
    "Competition in accelerated computing is intense and could reduce our margins.</p>"
    "<div>Item 7. Management's Discussion and Analysis</div>"
    "<p>Data center revenue grew sharply this fiscal year on strong demand for our platforms.</p>"
    "</body></html>"
)


def _meta(ticker: str, dtype: str, filing: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "document_type": dtype,
        "source": "SEC",
        "source_url": f"https://www.sec.gov/Archives/edgar/data/x/{ticker.lower()}-10k.htm",
        "filing_date": filing,
        "document_id": f"{ticker}:{dtype}:{filing}:0000000000-26-000001",
        "ingested_at": "2026-06-10T00:00:00+00:00",
    }


def _make_filing(documents_dir: Path, *, ticker: str = "NVDA", dtype: str = "10-K",
                 filing: str = "2026-02-25") -> None:
    d = documents_dir / "sec" / ticker / dtype / filing
    d.mkdir(parents=True)
    (d / "filing.html").write_text(_HTML, encoding="utf-8")
    (d / "metadata.json").write_text(json.dumps(_meta(ticker, dtype, filing)), encoding="utf-8")


def test_iter_filing_dirs_finds_downloaded(tmp_path: Path) -> None:
    _make_filing(tmp_path, dtype="10-K")
    _make_filing(tmp_path, dtype="10-Q", filing="2025-11-20")
    dirs = iter_filing_dirs(tmp_path, "NVDA")
    assert len(dirs) == 2 and all((d / "filing.html").exists() for d in dirs)


def test_iter_filing_dirs_empty_for_missing_ticker(tmp_path: Path) -> None:
    assert iter_filing_dirs(tmp_path, "TSLA") == []


def test_ingest_ticker_loads_chunks_into_store(tmp_path: Path) -> None:
    _make_filing(tmp_path)
    store = InMemoryVectorStore()
    result = ingest_ticker("NVDA", documents_dir=tmp_path, embedder=_EMB, store=store)
    assert isinstance(result, IngestResult)
    assert result.filings == 1 and result.chunks >= 2
    assert store.count() == result.chunks


def test_ingest_maintains_sparse_index_in_lockstep(tmp_path: Path) -> None:
    _make_filing(tmp_path)
    store = InMemoryVectorStore()
    sparse = InMemoryBM25Store()
    result = ingest_ticker(
        "NVDA", documents_dir=tmp_path, embedder=_EMB, store=store, sparse_store=sparse
    )
    # Both indexes hold the same chunks; the BM25 side finds a term from the filing text.
    assert sparse.count() == result.chunks == store.count()
    hits = sparse.search("export", top_k=5, where=ChunkFilter(ticker="NVDA"))
    assert hits  # the sparse index is queryable right after ingest


def test_incremental_ingest_backfills_sparse_without_re_embedding(tmp_path: Path) -> None:
    _make_filing(tmp_path)
    store = InMemoryVectorStore()
    # First: dense-only ingest (sparse index does not yet exist — simulates a pre-A3 corpus).
    ingest_ticker("NVDA", documents_dir=tmp_path, embedder=_EMB, store=store)
    sparse = InMemoryBM25Store()

    class _CountingEmbedder:
        name = "counting"
        dim = 32
        calls = 0

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            type(self).calls += 1
            return _EMB.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return _EMB.embed_query(text)

    emb = _CountingEmbedder()
    # Incremental re-ingest: dense chunks already present → 0 embed calls, but the empty sparse
    # index backfills from its OWN existing_ids (all chunks new to it).
    result = ingest_ticker(
        "NVDA", documents_dir=tmp_path, embedder=emb, store=store, sparse_store=sparse,
        incremental=True,
    )
    assert result.chunks == 0  # nothing new to embed
    assert emb.calls == 0  # so the embedder was never called
    assert sparse.count() == store.count() > 0  # yet the sparse index was fully backfilled


def test_backfill_sparse_indexes_without_embedding(tmp_path: Path) -> None:
    _make_filing(tmp_path)
    # Dense corpus exists but the sparse index is empty (simulates a pre-A3 corpus).
    store = InMemoryVectorStore()
    ingest_ticker("NVDA", documents_dir=tmp_path, embedder=_EMB, store=store)
    sparse = InMemoryBM25Store()
    assert sparse.count() == 0

    indexed = backfill_sparse(["NVDA"], documents_dir=tmp_path, sparse_store=sparse)

    assert indexed == store.count() > 0  # every dense chunk now in BM25 too — no embedder involved
    assert sparse.search("export", top_k=5)  # queryable immediately
    # Idempotent: a second backfill adds nothing.
    assert backfill_sparse(["NVDA"], documents_dir=tmp_path, sparse_store=sparse) == 0


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    _make_filing(tmp_path)
    store = InMemoryVectorStore()
    first = ingest_ticker("NVDA", documents_dir=tmp_path, embedder=_EMB, store=store)
    ingest_ticker("NVDA", documents_dir=tmp_path, embedder=_EMB, store=store)  # re-ingest
    assert store.count() == first.chunks  # upsert by chunk_id -> no duplication


def test_ingest_then_retrieve_end_to_end(tmp_path: Path) -> None:
    _make_filing(tmp_path)
    store = InMemoryVectorStore()
    ingest_ticker("NVDA", documents_dir=tmp_path, embedder=_EMB, store=store)
    # FakeEmbedder is hash-based (not semantic), so query with a stored chunk's exact text to
    # get a deterministic self-match — this exercises the full ingest→retrieve path + filter.
    chunks = chunk_filing(load_filing(iter_filing_dirs(tmp_path, "NVDA")[0]))
    target = chunks[0]
    ev = Retriever(_EMB, store).retrieve(target.text, top_k=3, where=ChunkFilter(ticker="NVDA"))
    assert not ev.is_empty
    assert ev.chunks[0].chunk.chunk_id == target.chunk_id  # exact chunk retrieved first
    assert all(c.chunk.ticker == "NVDA" for c in ev.chunks)


def test_ingest_empty_corpus_is_zero(tmp_path: Path) -> None:
    store = InMemoryVectorStore()
    result = ingest_ticker("NVDA", documents_dir=tmp_path, embedder=_EMB, store=store)
    assert result.filings == 0 and result.chunks == 0 and store.count() == 0


# ---- 9a: embedding spend guard ----------------------------------------------


class _CountingEmbedder(FakeEmbedder):
    """FakeEmbedder that records embed_documents calls — probes that the guard refuses
    *before* any embedding (i.e. no provider spend) when the ceiling is exceeded."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__(dim=dim)
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return super().embed_documents(texts)


def test_estimate_tokens_matches_word_proxy() -> None:
    # 6 words / 0.75 = 8 tokens (inverse of the chunking word-budget proxy).
    assert estimate_tokens(["one two three", "four five six"]) == 8
    assert estimate_tokens([]) == 0


def test_ingest_under_ceiling_succeeds_and_reports_tokens(tmp_path: Path) -> None:
    _make_filing(tmp_path)
    store = InMemoryVectorStore()
    result = ingest_ticker(
        "NVDA", documents_dir=tmp_path, embedder=_EMB, store=store, max_embed_tokens=10_000
    )
    assert result.chunks >= 2 and result.embed_tokens > 0
    assert store.count() == result.chunks


def test_ingest_over_ceiling_refuses_before_embedding(tmp_path: Path) -> None:
    _make_filing(tmp_path)
    store = InMemoryVectorStore()
    emb = _CountingEmbedder()
    with pytest.raises(EmbedBudgetExceeded) as ei:
        ingest_ticker("NVDA", documents_dir=tmp_path, embedder=emb, store=store, max_embed_tokens=1)
    assert emb.calls == 0  # refused BEFORE any embedding call — no spend
    assert store.count() == 0  # nothing persisted
    assert ei.value.ticker == "NVDA" and ei.value.ceiling == 1 and ei.value.estimated > 1


def test_ingest_no_ceiling_is_unlimited(tmp_path: Path) -> None:
    _make_filing(tmp_path)
    store = InMemoryVectorStore()
    # Default max_embed_tokens=None -> guard disabled, behaves exactly as before.
    result = ingest_ticker("NVDA", documents_dir=tmp_path, embedder=_EMB, store=store)
    assert store.count() == result.chunks and result.chunks >= 2


# ---- 9c-run hardening: bulk_ingest isolation + retry -------------------------

def _nosleep(_seconds: float) -> None:
    """Injected no-op backoff so retry tests don't actually wait."""


def test_bulk_ingest_aggregates_across_tickers(tmp_path: Path) -> None:
    _make_filing(tmp_path, ticker="NVDA")
    _make_filing(tmp_path, ticker="AVGO")
    store = InMemoryVectorStore()
    res = bulk_ingest(
        ["NVDA", "AVGO"], documents_dir=tmp_path, embedder=_EMB, store=store, sleep=_nosleep
    )
    assert isinstance(res, BulkIngestResult)
    assert res.tickers == 2 and not res.failed_tickers
    assert res.chunks == store.count() and res.chunks > 0
    assert {r.ticker for r in res.per_ticker} == {"NVDA", "AVGO"}


def test_bulk_ingest_isolates_failure_and_retries_transient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Drive bulk_ingest's loop directly: GOOD succeeds, FLAKY fails once then succeeds (retry),
    # BAD always fails (recorded, run continues). A transient blip must NOT abort the run.
    import stock_agent.rag.pipeline as pl

    calls = {"GOOD": 0, "FLAKY": 0, "BAD": 0}

    def fake_ingest(sym: str, **kw: object) -> IngestResult:
        calls[sym] += 1
        if sym == "BAD":
            raise RuntimeError("network down")
        if sym == "FLAKY" and calls["FLAKY"] == 1:
            raise RuntimeError("transient blip")  # recovers on retry
        return IngestResult(ticker=sym, filings=1, chunks=2, embed_tokens=10)

    monkeypatch.setattr(pl, "ingest_ticker", fake_ingest)
    res = bulk_ingest(
        ["GOOD", "FLAKY", "BAD"], documents_dir=tmp_path, embedder=_EMB,
        store=InMemoryVectorStore(), retries=2, sleep=_nosleep,
    )
    assert res.tickers == 3
    assert [r.ticker for r in res.per_ticker] == ["GOOD", "FLAKY"]  # BAD isolated, run continued
    assert len(res.failed_tickers) == 1 and res.failed_tickers[0].startswith("BAD:")
    assert calls["FLAKY"] == 2  # failed once, retried, succeeded
    assert calls["BAD"] == 3  # initial + 2 retries, then recorded


def test_incremental_ingest_embeds_only_new_chunks(tmp_path: Path) -> None:
    # 9e: a refresh must re-embed only NEW chunks, never the whole ticker again.
    _make_filing(tmp_path, dtype="10-K", filing="2025-02-26")
    store = InMemoryVectorStore()
    emb = _CountingEmbedder()
    inc = {"documents_dir": tmp_path, "embedder": emb, "store": store, "incremental": True}
    first = ingest_ticker("NVDA", **inc)  # type: ignore[arg-type]
    assert first.chunks >= 2 and first.skipped_existing == 0
    assert emb.calls == 1  # embedded once

    # Re-ingest with nothing new -> all chunks already present -> no embedding call at all.
    again = ingest_ticker("NVDA", **inc)  # type: ignore[arg-type]
    assert again.chunks == 0 and again.skipped_existing == first.chunks
    assert emb.calls == 1  # unchanged — no re-embed
    assert again.embed_tokens == 0

    # Add a NEW filing -> only its chunks are embedded; the old ones are skipped.
    _make_filing(tmp_path, dtype="10-Q", filing="2025-05-20")
    delta = ingest_ticker("NVDA", **inc)  # type: ignore[arg-type]
    assert delta.chunks >= 1 and delta.skipped_existing == first.chunks
    assert emb.calls == 2  # embedded exactly once more (only the new chunks)
    assert store.count() == first.chunks + delta.chunks


def test_bulk_ingest_incremental_passes_through(tmp_path: Path) -> None:
    _make_filing(tmp_path, ticker="NVDA")
    store = InMemoryVectorStore()
    bulk_ingest(["NVDA"], documents_dir=tmp_path, embedder=_EMB, store=store, sleep=_nosleep)
    # Second pass incremental -> everything already present, nothing re-embedded.
    res = bulk_ingest(
        ["NVDA"], documents_dir=tmp_path, embedder=_EMB, store=store,
        incremental=True, sleep=_nosleep,
    )
    assert res.chunks == 0 and res.skipped_existing == store.count() and res.skipped_existing > 0


def test_bulk_ingest_budget_exceeded_aborts_not_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import stock_agent.rag.pipeline as pl

    def fake_ingest(sym: str, **kw: object) -> IngestResult:
        raise EmbedBudgetExceeded(sym, 100, 10)  # deliberate spend stop

    monkeypatch.setattr(pl, "ingest_ticker", fake_ingest)
    with pytest.raises(EmbedBudgetExceeded):  # propagates — NOT caught/retried as transient
        bulk_ingest(
            ["NVDA"], documents_dir=tmp_path, embedder=_EMB,
            store=InMemoryVectorStore(), sleep=_nosleep,
        )
