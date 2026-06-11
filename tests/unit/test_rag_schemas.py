"""RAG P0: document/retrieval schema conformance + settings defaults (no I/O)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from stock_agent.schemas.documents import Document, DocumentChunk, DocumentMetadata
from stock_agent.schemas.retrieval import EvidenceSet, RetrievedChunk
from stock_agent.settings import Settings


def _meta(section: str | None = None) -> DocumentMetadata:
    return DocumentMetadata(
        ticker="NVDA",
        document_type="10-K",
        source="SEC",
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/x.htm",
        filing_date=date(2025, 2, 26),
        document_id="NVDA:10-K:2025-02-26:0001045810-25-000017",
        section=section,
        ingested_at=datetime(2026, 6, 9, tzinfo=UTC),
    )


def test_document_roundtrips_json() -> None:
    doc = Document(metadata=_meta(), text="Risk factors and MD&A...")
    assert Document.model_validate(doc.model_dump(mode="json")) == doc


def test_chunk_from_metadata_copies_provenance_and_builds_id() -> None:
    chunk = DocumentChunk.from_metadata(_meta(), chunk_index=3, text="...", section="Item 1A")
    assert chunk.chunk_id == "NVDA:10-K:2025-02-26:0001045810-25-000017:3"
    assert chunk.document_id == _meta().document_id
    assert chunk.ticker == "NVDA" and chunk.document_type == "10-K"
    assert chunk.section == "Item 1A"  # explicit section overrides the metadata's


def test_chunk_inherits_metadata_section_when_not_overridden() -> None:
    chunk = DocumentChunk.from_metadata(_meta(section="Item 7. MD&A"), chunk_index=0, text="x")
    assert chunk.section == "Item 7. MD&A"


def test_document_type_literal_is_enforced() -> None:
    # Construct directly (model_copy skips validation); "DEF 14A" is not in the MVP corpus.
    with pytest.raises(ValidationError):
        DocumentMetadata(
            ticker="NVDA",
            document_type="DEF 14A",
            source="SEC",
            source_url="https://www.sec.gov/x.htm",
            filing_date=date(2025, 2, 26),
            document_id="id",
            ingested_at=datetime(2026, 6, 9, tzinfo=UTC),
        )


def test_retrieved_chunk_citation_label() -> None:
    meta = _meta(section="Item 1A. Risk Factors")
    chunk = DocumentChunk.from_metadata(meta, chunk_index=0, text="x")
    rc = RetrievedChunk(chunk=chunk, score=0.81)
    assert rc.citation_label() == "NVDA 10-K Feb 26, 2025 — Item 1A. Risk Factors"
    bare = RetrievedChunk(
        chunk=DocumentChunk.from_metadata(_meta(), chunk_index=0, text="x"), score=0.5
    )
    assert bare.citation_label() == "NVDA 10-K Feb 26, 2025"  # no section -> no suffix


def test_evidence_set_empty_and_allowed_ids() -> None:
    empty = EvidenceSet(query="What drove revenue?")
    assert empty.is_empty and len(empty) == 0 and empty.allowed_chunk_ids() == set()

    c0 = DocumentChunk.from_metadata(_meta(), chunk_index=0, text="a")
    c1 = DocumentChunk.from_metadata(_meta(), chunk_index=1, text="b")
    ev = EvidenceSet(
        query="q",
        chunks=[RetrievedChunk(chunk=c0, score=0.9), RetrievedChunk(chunk=c1, score=0.7)],
    )
    assert not ev.is_empty and len(ev) == 2
    assert ev.allowed_chunk_ids() == {c0.chunk_id, c1.chunk_id}


# ---- settings defaults (P0 config) -------------------------------------------
def test_rag_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.embedding_provider == "local"
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.rag_top_k == 8
    assert 0.0 <= s.rag_chunk_overlap < 1.0
    assert s.documents_dir == Path("data/raw")
    assert s.vector_store_dir == Path("data/vectorstore")
    assert s.openai_api_key is None and s.sec_user_agent is None
