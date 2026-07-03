"""AgentEvent — the streaming event schema for the agent runtime (Phase 2, P2.1).

A discriminated union yielded by :func:`stock_agent.agent.runtime.run_agent_events` (and,
in later slices, the router and the FastAPI SSE adapter). Each event is JSON-serializable via
``to_wire()`` — that dict is the exact SSE frame the React client consumes (plan §4).

Ownership / dependency direction (why the union is split):
- The **runtime** emits only the events it can produce without importing presentation code:
  ``tool_start``, ``tool_finish``, ``token``, ``final``, ``error``. It must never import
  ``ui/`` or ``viz/`` — ``ui/`` (Streamlit) imports ``agent/``, so importing back would cycle.
- ``turn_start`` / ``route_decided`` come from the **router** (P2.2); ``tiles`` / ``chart`` /
  ``sources`` are built by the **api adapter** (P2.2) from the same ``ToolInvocation`` list via
  the pure builders (``ui.tiles.stat_tiles_from_tool_results``, ``viz.charts.charts_for``,
  ``ui.state.sources_from_tool_results``). Defining the full union here keeps those slices from
  redefining the schema.

Wire vs. in-process: most events serialize 1:1. The exception is :class:`Final`, which carries
``messages`` / ``tool_results`` in-process (so the synchronous drain ``run_agent`` can rebuild an
``AgentResult``) but projects to the §4 subset ``{turn_id, tool_calls, iterations, grounded}`` on
the wire — the answer text reaches the client as ``token`` deltas, never re-sent on ``final``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from zlib import crc32

if TYPE_CHECKING:  # avoid an import cycle: runtime imports events, not vice versa at runtime.
    from stock_agent.agent.runtime import ToolInvocation


def hue_for(tool: str) -> int:
    """Stable per-tool hue key = ``crc32`` of the tool name (deterministic across processes).

    The client maps this to a palette slot (``hue_key % n_hues``); the palette size is a
    presentation detail the runtime must not know, so we emit the raw crc32 int, not a hue
    token. ``crc32`` (unlike salted ``hash()``) is stable across runs/processes → unit-testable
    and gives a given tool the same color every turn. Mirrors ``ui.html.tool_hue``'s convention.
    """
    return crc32(tool.encode("utf-8"))


@dataclass(frozen=True)
class TurnStart:
    """A turn begins. Emitted by the router (carries the resolved routing context)."""

    thread_id: str
    turn_id: str
    route: str
    ticker: str | None = None
    type: str = field(default="turn_start", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "route": self.route,
            "ticker": self.ticker,
        }


@dataclass(frozen=True)
class RouteDecided:
    """The router picked a path. ``mode`` is ``"deterministic"`` or ``"auto"`` (plan §4)."""

    mode: str
    route_name: str
    note: str = ""
    type: str = field(default="route_decided", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "mode": self.mode,
            "route_name": self.route_name,
            "note": self.note,
        }


@dataclass(frozen=True)
class ToolStart:
    """A tool is about to execute. ``input_summary`` is a compact trace-chip label."""

    tool: str
    input_summary: str
    hue_key: int
    type: str = field(default="tool_start", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tool": self.tool,
            "input_summary": self.input_summary,
            "hue_key": self.hue_key,
        }


@dataclass(frozen=True)
class ToolFinish:
    """A tool finished. ``ok`` mirrors the runtime's error convention (``"error"`` key absent)."""

    tool: str
    ok: bool
    elapsed_ms: float
    type: str = field(default="tool_finish", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tool": self.tool,
            "ok": self.ok,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass(frozen=True)
class Tiles:
    """Headline stat tiles (from ``ui.tiles.stat_tiles_from_tool_results``); adapter-built."""

    tiles: list[dict[str, str]]
    type: str = field(default="tiles", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type, "tiles": self.tiles}


@dataclass(frozen=True)
class Chart:
    """A chart spec (``viz.charts.ChartSpec.to_dict()``). Built by the adapter."""

    spec: dict[str, Any]
    type: str = field(default="chart", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type, "spec": self.spec}


@dataclass(frozen=True)
class Token:
    """A candidate-text delta from the LLM. In P2.1 a single delta carries the whole answer;
    P2.4 (true streaming) yields many. Provisional until ``final`` (plan §5)."""

    text: str
    type: str = field(default="token", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text}


@dataclass(frozen=True)
class Sources:
    """SEC citations (from ``ui.state.sources_from_tool_results``). Built by the adapter."""

    citations: list[dict[str, Any]]
    type: str = field(default="sources", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type, "citations": self.citations}


@dataclass(frozen=True)
class Final:
    """Terminal success — emitted only AFTER the numeric-grounding guard passes (``grounded=True``).

    In-process it also carries ``messages`` (full transcript for next-turn history) and
    ``tool_results`` (structured invocations for charting) so the synchronous drain can rebuild an
    ``AgentResult``; neither is sent on the wire (the answer text already streamed as ``token``s).
    """

    tool_calls: list[str]
    iterations: int
    messages: list[dict[str, Any]]
    tool_results: list[ToolInvocation]
    turn_id: str | None = None
    grounded: bool = True
    type: str = field(default="final", init=False)

    def to_wire(self) -> dict[str, Any]:
        # Wire subset per §4 — text is delivered via `token`s, transcript stays server-side.
        return {
            "type": self.type,
            "turn_id": self.turn_id,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "grounded": self.grounded,
        }


@dataclass(frozen=True)
class Error:
    """Terminal failure. ``code`` is a stable machine tag (``grounding`` | ``max_iterations`` |
    ``agent``); the drain maps it back to the matching exception (plan §5: error, not final)."""

    code: str
    message: str
    type: str = field(default="error", init=False)

    def to_wire(self) -> dict[str, Any]:
        return {"type": self.type, "code": self.code, "message": self.message}


# Discriminated union (the ``type`` field is the discriminant). Ordered per the §5 stream contract.
AgentEvent = (
    TurnStart
    | RouteDecided
    | ToolStart
    | ToolFinish
    | Tiles
    | Chart
    | Token
    | Sources
    | Final
    | Error
)
