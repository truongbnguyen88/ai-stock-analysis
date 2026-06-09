"""Retrieval domain models for the RAG layer.

A ``RetrievedChunk`` is a ``DocumentChunk`` plus its similarity score; an
``EvidenceSet`` is the ranked result for one query — the only thing handed to the
synthesis LLM. Keeping the chunk intact means every cited fact traces back to a
real source document (the citation guard checks answers against this set).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from stock_agent.schemas.documents import DocumentChunk


class RetrievedChunk(BaseModel):
    """A chunk returned by vector search, with its relevance score (higher = closer)."""

    chunk: DocumentChunk
    score: float

    def citation_label(self) -> str:
        """Human-readable source label, e.g. 'NVDA 10-K 2025-02-26 — Item 1A. Risk Factors'."""
        c = self.chunk
        base = f"{c.ticker} {c.document_type} {c.filing_date.isoformat()}"
        return f"{base} — {c.section}" if c.section else base


class EvidenceSet(BaseModel):
    """Ranked retrieved evidence for one question (the LLM's entire grounding)."""

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        """True when retrieval found nothing (synthesis must then refuse, not invent)."""
        return not self.chunks

    def allowed_chunk_ids(self) -> set[str]:
        """Chunk ids the answer may cite (the citation guard's allow-set)."""
        return {rc.chunk.chunk_id for rc in self.chunks}
