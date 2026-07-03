"""Pure display-state transforms for the Streamlit frontend (redesign R1).

No Streamlit, no I/O — mirrors ``stock_agent.ui.capabilities``/``theme`` so it is
type-checked and unit-tested like the rest of the package. The repo-root ``ui/`` view
and session modules call these to (de)serialize a chat thread's display messages for the
``ChatStore`` and to collect SEC citations from tool results.

Extracted verbatim from ``ui/chat_app.py`` (R1 refactor) — behavior-preserving.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from stock_agent.viz.charts import ChartSpec


def serialize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Display messages with ``ChartSpec`` objects flattened to dicts (for the store).

    Charts/sources keys are omitted when empty, matching the on-disk shape prior to R1.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        sm: dict[str, Any] = {"role": m["role"], "content": m["content"]}
        if m.get("charts"):
            sm["charts"] = [c.to_dict() for c in m["charts"]]
        if m.get("sources"):
            sm["sources"] = m["sources"]  # plain dicts already; survive a restart
        # R4: stat tiles are display-ready plain dicts — persist so history re-renders
        # them intact after a rerun/restart (like charts/sources; the tool-trace chip row
        # stays ephemeral/live-turn-only, matching the pre-R4 "tools used" caption).
        if m.get("tiles"):
            sm["tiles"] = m["tiles"]
        out.append(sm)
    return out


def deserialize_messages(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inverse of :func:`serialize_messages` — rebuild ``ChartSpec`` objects for render."""
    out: list[dict[str, Any]] = []
    for m in raw:
        out.append(
            {
                "role": m["role"],
                "content": m["content"],
                "charts": [ChartSpec.from_dict(c) for c in m.get("charts", [])],
                "sources": m.get("sources", []),
                # R4: tolerant of older threads that predate this key (default empty).
                "tiles": m.get("tiles", []),
            }
        )
    return out


class _HasResult(Protocol):
    """Structural view of a tool call (matches ``agent.runtime.ToolInvocation``)."""

    result: dict[str, Any]


def sources_from_tool_results(tool_results: Sequence[_HasResult]) -> list[dict[str, Any]]:
    """Collect SEC citations from RAG tools (search_filings / research_multistep / summary).

    Tool-name agnostic: pulls any ``citations`` field from any tool result, so new RAG
    tools surface automatically. Citations come from the tool OUTPUT (P7/P8 resolved them
    against the retrieved set), not the LLM — so surfacing them here can't introduce a
    fabricated source. Deduped by ``(marker, label)``, preserving first-seen order.
    """
    seen: set[tuple[Any, str]] = set()
    sources: list[dict[str, Any]] = []
    for inv in tool_results:
        cits = inv.result.get("citations", []) if isinstance(inv.result, dict) else []
        for cit in cits:
            key = (cit.get("marker"), cit.get("label", ""))
            if key not in seen:
                seen.add(key)
                sources.append({"marker": cit.get("marker"), "label": cit.get("label", "")})
    return sources
