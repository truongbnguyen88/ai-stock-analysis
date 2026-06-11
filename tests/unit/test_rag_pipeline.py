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
    EmbedBudgetExceeded,
    IngestResult,
    ingest_ticker,
    iter_filing_dirs,
)
from stock_agent.rag.retriever import Retriever
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
