"""Shared helpers for the research package's grounded synthesis calls (P7 + P8).

Both the QA synthesis (``synthesis.py``) and the memo synthesis (``memo.py``) parse a JSON
object out of the model's reply, scan it for inline ``[n]`` citation markers, and build a
corrective retry note. These live here — rather than one synthesis module importing the
other's privates — so the two paths share one implementation.
"""

from __future__ import annotations

import json
import re
from typing import Any

_MARKER = re.compile(r"\[(\d+)\]")  # inline citation markers, e.g. "[2]"


def loads_lenient(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating stray prose around it (models sometimes wrap it)."""
    try:
        result: Any = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("model reply was not a JSON object")
    return result


def markers_in_text(text: str) -> set[int]:
    """Source numbers cited inline in the text (the real citation surface), e.g. {2, 5}."""
    return {int(m.group(1)) for m in _MARKER.finditer(text)}


def correction(
    bad_cites: list[int], ungrounded: list[str], n: int, *, allow_insufficient: bool = True
) -> str:
    """A corrective retry note naming exactly which citations/figures to fix.

    ``allow_insufficient`` keeps the "or set insufficient_evidence" escape clause for the QA
    path; the memo has no such field, so it passes ``False``.
    """
    problems: list[str] = []
    if bad_cites:
        problems.append(f"you cited sources {bad_cites}, but only [1]–[{n}] exist")
    if ungrounded:
        problems.append(f"these figures are not in any source: {', '.join(ungrounded[:5])}")
    guidance = "Use ONLY the provided sources and only numbers that appear in them"
    if allow_insufficient:
        guidance += ' — or set "insufficient_evidence": true'
    guidance += ". Do not invent sources or figures."
    return "\n\nIMPORTANT: " + "; ".join(problems) + ". " + guidance
