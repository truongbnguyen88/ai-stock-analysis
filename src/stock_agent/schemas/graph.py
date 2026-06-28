"""Knowledge-graph domain models (advanced-RAG A5) — typed entities and provenance-bearing edges.

A5 builds a lightweight entity–relationship graph over the SEC corpus so that *structural* and
*multi-hop bridging* questions (e.g. "who are NVDA's suppliers", "does NVDA's memory supplier face
the same capacity risk") become a graph **traversal** instead of something the embedder must
accidentally encode in a dot product. Theory → [docs/rag_concepts.md](../../../docs/rag_concepts.md)
§16; build log → [docs/rag_implementation_notes.md](../../../docs/rag_implementation_notes.md) §A5.

Two shapes, deliberately tiny (MVP scope guard):
- ``Entity`` — a typed node. A **company** node's canonical id is its **ticker** (``MU``), resolved
  once at extraction time; non-company nodes use a normalized-name id.
- ``Edge`` — a typed ``(subject, relation, object)`` triple that **always carries provenance**: the
  ``chunk_id``(s) it was extracted from, so every graph answer remains citeable (the edge analogue
  of the P7 citation guard). An edge with no provenance is inadmissible (``min_length=1``).
"""

from __future__ import annotations

from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, Field

# Entity types (MVP, 5). NB: supplier/customer/competitor are *relations*, not types.
EntityType = Literal["company", "product", "segment", "risk", "regulatory_topic"]

# Relations (MVP, 4). Direction matters (depends_on is asymmetric); supplies_to is the inverse of
# depends_on, derived on traversal rather than stored.
Relation = Literal["depends_on", "competes_with", "mentions_risk", "exposed_to"]

# The object-entity type implied by each relation (subject is always the filing's company). Deriving
# the object type from the relation — rather than asking the LLM for it — removes a failure mode
# (an inconsistent type label) and keeps extraction's output minimal.
RELATION_OBJECT_TYPE: dict[Relation, EntityType] = {
    "depends_on": "company",
    "competes_with": "company",
    "mentions_risk": "risk",
    "exposed_to": "regulatory_topic",
}


class Entity(BaseModel):
    """A typed graph node. ``id`` is the canonical key; ``name`` is the surface form as written.

    For a **company** the canonical ``id`` is its ticker (e.g. ``"MU"``) and ``ticker`` repeats it;
    when a company name cannot be resolved to a ticker, ``id`` falls back to the normalized name and
    ``ticker`` is ``None`` (lower-value but still citeable). Non-company nodes (``risk`` /
    ``regulatory_topic`` / ``product`` / ``segment``) use a normalized-name id and ``ticker=None``.
    """

    id: str  # canonical key: ticker for resolved companies ("MU"); normalized name otherwise
    name: str  # surface form as written in the filing ("Micron")
    type: EntityType
    ticker: str | None = None  # resolved ticker for company entities, else None


class Edge(BaseModel):
    """A provenance-bearing ``(subject, relation, object)`` triple extracted from a filing.

    ``subject`` and ``object`` are entity ids (``subject`` is the filing's own company, e.g.
    ``"NVDA"``; ``object`` is the related entity id, e.g. ``"MU"`` or a risk node). ``provenance``
    is the non-empty list of ``chunk_id``s the triple was stated in — the citation anchor that keeps
    graph answers grounded. ``filing_date`` / ``source_url`` come from the provenance chunk;
    ``confidence`` is the extractor's self-reported score (gated by ``graph_min_confidence``).
    """

    subject: str  # entity id — the filing's company (e.g. "NVDA")
    relation: Relation
    object: str  # entity id — the related entity (e.g. "MU", or a risk-node id)
    provenance: list[str] = Field(min_length=1)  # chunk_ids the triple was extracted from
    filing_date: Date
    source_url: str
    confidence: float = 1.0
