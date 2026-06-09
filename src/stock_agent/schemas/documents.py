"""Document domain models for the RAG layer (SEC-grounded research).

The normalized shapes that flow ingestion → retrieval: a parsed ``Document`` (full
text + provenance), and the ``DocumentChunk``s it is split into. Chunks carry FLAT
metadata (ticker, type, date, section, url) so the vector store can filter on them
at retrieval without re-parsing — mirroring how providers normalize to ``schemas/``
at the boundary so business logic never touches raw filings.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# MVP corpus = SEC filings; widen (transcript/presentation/news/internal) in V1.
DocumentType = Literal["10-K", "10-Q", "8-K"]
DocumentSource = Literal["SEC"]


class DocumentMetadata(BaseModel):
    """Provenance for one source document — the basis for every citation."""

    ticker: str
    document_type: DocumentType
    source: DocumentSource
    source_url: str
    filing_date: Date  # the filing's as-of date (point-in-time anchor for future model use)
    document_id: str  # stable id, e.g. f"{ticker}:{document_type}:{filing_date}:{accession}"
    section: str | None = None  # section label when known (e.g. "Item 1A. Risk Factors")
    ingested_at: datetime


class Document(BaseModel):
    """A parsed source document: normalized full text plus its provenance."""

    metadata: DocumentMetadata
    text: str


class DocumentChunk(BaseModel):
    """One retrieval unit: a slice of a document with flat, filterable metadata.

    Metadata is duplicated onto each chunk (denormalized) on purpose: the vector
    store filters on these scalar fields (ticker/type/date/section/source) at query
    time, and a citation is reconstructable from a chunk alone — no document join.
    """

    chunk_id: str  # unique, e.g. f"{document_id}:{chunk_index}"
    document_id: str
    chunk_index: int = Field(ge=0)  # position within the source document
    text: str

    # Flat provenance copied from DocumentMetadata (for filtering + citation).
    ticker: str
    document_type: DocumentType
    source: DocumentSource
    source_url: str
    filing_date: Date
    section: str | None = None

    @classmethod
    def from_metadata(
        cls, meta: DocumentMetadata, *, chunk_index: int, text: str, section: str | None = None
    ) -> DocumentChunk:
        """Build a chunk, copying provenance from ``meta`` (section may override)."""
        return cls(
            chunk_id=f"{meta.document_id}:{chunk_index}",
            document_id=meta.document_id,
            chunk_index=chunk_index,
            text=text,
            ticker=meta.ticker,
            document_type=meta.document_type,
            source=meta.source,
            source_url=meta.source_url,
            filing_date=meta.filing_date,
            section=section if section is not None else meta.section,
        )
