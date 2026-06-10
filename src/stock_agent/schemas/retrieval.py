"""Retrieval domain models for the RAG layer.

A ``RetrievedChunk`` is a ``DocumentChunk`` plus its similarity score; an
``EvidenceSet`` is the ranked result for one query — the only thing handed to the
synthesis LLM. Keeping the chunk intact means every cited fact traces back to a
real source document (the citation guard checks answers against this set).
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field

from stock_agent.schemas.documents import DocumentChunk, DocumentType


class ChunkFilter(BaseModel):
    """Metadata predicate for a vector-store query.

    Set fields are AND-combined; an unset (``None``) field imposes no constraint.
    The vector store applies this *before* ranking (filter then top-k), so retrieval
    is scoped to e.g. one ticker / form / section / date window. ``date_from`` and
    ``date_to`` are inclusive bounds on ``filing_date``.
    """

    ticker: str | None = None
    document_types: list[DocumentType] | None = None
    sections: list[str] | None = None
    date_from: Date | None = None
    date_to: Date | None = None

    @property
    def is_empty(self) -> bool:
        """True when no constraint is set (the query spans the whole store)."""
        return not any(
            (self.ticker, self.document_types, self.sections, self.date_from, self.date_to)
        )

    def matches(self, chunk: DocumentChunk) -> bool:
        """Whether ``chunk`` satisfies every set constraint (used by in-Python backends)."""
        return (
            (self.ticker is None or chunk.ticker == self.ticker)
            and (self.document_types is None or chunk.document_type in self.document_types)
            and (self.sections is None or chunk.section in self.sections)
            and (self.date_from is None or chunk.filing_date >= self.date_from)
            and (self.date_to is None or chunk.filing_date <= self.date_to)
        )


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
