"""Research-answer domain models (RAG P7).

A ``GroundedAnswer`` is the output of the single SEC-grounded synthesis call: a cited
answer whose every source marker resolves to a chunk that was actually retrieved (the
citation guard enforces this). ``insufficient_evidence`` is the honest-refusal signal —
when retrieval found nothing relevant, the answer is "Insufficient evidence found." and
no claim is made.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Resolves an inline ``[n]`` marker in the answer to the chunk it cites."""

    marker: int  # the [n] used inline in the answer text
    chunk_id: str  # the retrieved chunk this marker refers to (⊆ the EvidenceSet)
    label: str  # human-readable, e.g. "NVDA 10-K Feb 26, 2025 — Item 1A. Risk Factors"


class GroundedAnswer(BaseModel):
    """A SEC-grounded answer to one question, with resolved citations."""

    question: str
    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False
