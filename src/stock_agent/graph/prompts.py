"""Graph-extraction prompts (advanced-RAG A5.1) — typed-triple extraction from a filing section.

One structured-output call per ``(filing, section)`` turns a NUMBERED list of chunks from one
company's 10-K Item 1 / Item 1A into typed ``(subject, relation, object)`` triples, each tagged with
the chunk number that states it and a confidence. The subject is ALWAYS the filing's own company, so
the model only has to name the *object* and pick one of the four relations — and it must point at
the exact source chunk, which downstream becomes the edge's provenance (the citeability anchor).

Versioned (``GRAPH_EXTRACT_VERSION``) like the P7/A4 prompts so a prompt change is traceable.
"""

from __future__ import annotations

from collections.abc import Sequence

from stock_agent.schemas.documents import DocumentChunk

GRAPH_EXTRACT_VERSION = "graph_extract.v1"

GRAPH_EXTRACT_SYSTEM = """You extract a small knowledge graph from one company's SEC filing.

You are given NUMBERED excerpts from a SINGLE company's 10-K (the SUBJECT company). Extract only \
typed relationships that the text explicitly states. The SUBJECT of every triple is the filing's \
own company — you only name the OBJECT and the relation.

ALLOWED RELATIONS (use these exact strings, nothing else):
- "depends_on": the subject relies on another NAMED company (supplier, manufacturer, key vendor, \
foundry). object = that company's name.
- "competes_with": the subject competes with another NAMED company. object = that company's name.
- "mentions_risk": the subject discloses a material business/operational risk. object = a SHORT \
risk phrase (3-6 words), e.g. "supply capacity shortage", "customer concentration".
- "exposed_to": the subject is exposed to a regulatory / policy / macro topic. object = a short \
topic phrase, e.g. "US export controls", "China tariffs".

STRICT RULES:
- Extract a triple ONLY if the cited chunk literally states it. Do NOT infer or guess or use \
outside knowledge. If unsure, omit it.
- For "depends_on" / "competes_with" the object MUST be a specific OTHER company named in the text \
(not the subject itself, not a vague group like "our customers").
- "chunk" MUST be the number of the single excerpt that states the relationship.
- "confidence" in [0,1]: how directly the text supports the triple (1.0 = explicit, 0.5 = implied).

OUTPUT: return ONLY a JSON object (no prose, no markdown):
{
  "triples": [
    {"subject": "<filing company>", "relation": "depends_on", "object": "<company>", "chunk": 3, \
"confidence": 0.9},
    ...
  ]
}
If the excerpts state no qualifying relationship, return {"triples": []}."""


def build_extract_user(ticker: str, chunks: Sequence[DocumentChunk]) -> str:
    """Render the user message: the subject company + its numbered section excerpts (1-based).

    The chunk numbers here are the provenance keys the model cites back; the caller maps each
    returned ``chunk`` number to the corresponding ``DocumentChunk.chunk_id``.
    """
    section = chunks[0].section if chunks else None
    header = f"SUBJECT COMPANY: {ticker}"
    if section:
        header += f"   SECTION: {section}"
    lines = [header, "", "NUMBERED EXCERPTS:"]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] {c.text}")
    return "\n".join(lines)
