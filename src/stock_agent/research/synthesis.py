"""SEC-grounded question answering (RAG P7) — the single synthesis call + guards.

One LLM call turns a question + a retrieved ``EvidenceSet`` into a cited ``GroundedAnswer``.
Two guards enforce the project invariants at the output boundary:

- **Citation guard** — every source the answer cites (inline ``[n]`` *and* the ``citations``
  list) must be one of the numbered sources that were actually retrieved. A marker outside
  ``[1..N]`` is a fabricated citation.
- **Number grounding** — every percentage/decimal figure in the answer must appear in the
  source texts (reuses ``NumberGrounding``); the model may quote the filings but not invent
  or compute figures of its own.

On a violation we send one corrective retry, then raise rather than emit an ungrounded answer.
Empty retrieval short-circuits to "Insufficient evidence found." with NO LLM call (cost +
the honest-refusal invariant).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from stock_agent.llm.client import TextLLM
from stock_agent.llm.guards import NumberGrounding
from stock_agent.logging_config import get_logger
from stock_agent.research.prompts import SYSTEM, build_user
from stock_agent.schemas.research import GroundedAnswer, SourceCitation
from stock_agent.schemas.retrieval import EvidenceSet

log = get_logger(__name__)

_INSUFFICIENT = "Insufficient evidence found."
_MAX_TOKENS = 1200
_MARKER = re.compile(r"\[(\d+)\]")  # inline citation markers, e.g. "[2]"


class ResearchGuardError(RuntimeError):
    """Raised when the answer still cites fabricated sources / figures after a retry."""


class _RawAnswer(BaseModel):
    """The raw LLM JSON shape before citation markers are resolved to chunks."""

    answer: str = ""
    citations: list[int] = Field(default_factory=list)
    insufficient_evidence: bool = False


def _loads_lenient(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating stray prose around it (matches the summarizer)."""
    try:
        result: Any = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("research answer was not a JSON object")
    return result


def _markers_in_text(text: str) -> set[int]:
    """Source numbers cited inline in the answer (the real citation surface)."""
    return {int(m.group(1)) for m in _MARKER.finditer(text)}


def _correction(bad_cites: list[int], ungrounded: list[str], n: int) -> str:
    """Append a corrective note describing exactly what to fix."""
    problems: list[str] = []
    if bad_cites:
        problems.append(f"you cited sources {bad_cites}, but only [1]–[{n}] exist")
    if ungrounded:
        problems.append(f"these figures are not in any source: {', '.join(ungrounded[:5])}")
    return (
        "\n\nIMPORTANT: " + "; ".join(problems) + ". Use ONLY the provided sources and only "
        "numbers that appear in them — or set \"insufficient_evidence\": true. "
        "Do not invent sources or figures."
    )


def answer_question(
    question: str,
    evidence: EvidenceSet,
    *,
    llm: TextLLM,
    max_tokens: int = _MAX_TOKENS,
) -> GroundedAnswer:
    """Answer ``question`` strictly from ``evidence`` (one guarded LLM call)."""
    if evidence.is_empty:
        # No retrieved evidence -> refuse honestly, without paying for an LLM call.
        return GroundedAnswer(question=question, answer=_INSUFFICIENT, insufficient_evidence=True)

    grounding = NumberGrounding()
    for rc in evidence.chunks:
        grounding.add_from(rc.chunk.text)  # only figures present in the sources are allowed

    user = build_user(question, evidence.chunks)
    raw = llm.complete_json(system=SYSTEM, user=user, max_tokens=max_tokens)
    return _guarded(question, evidence, grounding, raw, user, llm, max_tokens, retry=True)


def _guarded(
    question: str,
    evidence: EvidenceSet,
    grounding: NumberGrounding,
    raw: str,
    user: str,
    llm: TextLLM,
    max_tokens: int,
    *,
    retry: bool,
) -> GroundedAnswer:
    parsed = _RawAnswer.model_validate(_loads_lenient(raw))
    if parsed.insufficient_evidence:
        return GroundedAnswer(question=question, answer=_INSUFFICIENT, insufficient_evidence=True)

    n = len(evidence.chunks)
    cited = set(parsed.citations) | _markers_in_text(parsed.answer)
    bad_cites = sorted(m for m in cited if not 1 <= m <= n)  # citations outside the retrieved set
    ungrounded = grounding.ungrounded(parsed.answer)  # figures not present in any source

    if bad_cites or ungrounded:
        if retry:
            log.warning(
                "research.guard.retry", bad_citations=bad_cites, ungrounded=ungrounded[:5]
            )
            corrected = llm.complete_json(
                system=SYSTEM, user=user + _correction(bad_cites, ungrounded, n),
                max_tokens=max_tokens,
            )
            return _guarded(question, evidence, grounding, corrected, user, llm, max_tokens,
                            retry=False)
        raise ResearchGuardError(
            f"answer cited invalid sources {bad_cites} / ungrounded figures {ungrounded[:5]}"
        )

    citations = [
        SourceCitation(
            marker=m,
            chunk_id=evidence.chunks[m - 1].chunk.chunk_id,
            label=evidence.chunks[m - 1].citation_label(),
        )
        for m in sorted(cited)
    ]
    return GroundedAnswer(question=question, answer=parsed.answer, citations=citations)
