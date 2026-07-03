"""AgentEvent → SSE adapter for ``POST /chat/stream`` (Phase 2, P2.2).

The router/runtime emit the events they *own* (``turn_start``, ``route_decided``,
``tool_start``/``tool_finish``, ``token``, ``final``/``error``). This layer weaves in the three
**presentation** events the plan assigns to the api adapter — ``tiles``, ``chart``, ``sources`` —
built by the *pure* builders over the turn's ``ToolInvocation`` list
(``ui.tiles.stat_tiles_from_tool_results``, ``viz.charts.charts_for``,
``ui.state.sources_from_tool_results``). Only the api layer may import ``ui``/``viz`` (nothing
imports ``api``, so no cycle; the runtime/router must not, or they'd cycle with Streamlit's ``ui``).

Ordering (plan §5): ``turn_start → route_decided → (tool_start, tool_finish)* → tiles → chart* →
token* → sources → final`` (or ``error``). Because tiles/chart/sources are functions of the *whole*
invocation list, they can only be built once tools are done — so this adapter **buffers the
token(s)** and flushes ``tiles → chart* → token* → sources → final`` together at ``Final``. Since
P2.4 the runtime streams the answer as MULTIPLE ``token`` deltas (``AnthropicToolClient.stream``);
they are emitted only *after* the grounding guard clears, so buffering them here to keep tiles/chart
first (§5) costs no latency (the whole answer is already assembled before the first delta). Live-
*during*-generation streaming (flush tiles/chart early, forward deltas as the model types) is
deferred: it would require a provisional-reset frame + a §5 relaxation and would surface ungrounded
figures pre-guard — see ``run_agent_events`` for the grounding-safe rationale.

On ``Error`` (grounding rejection after retry, or a tool/LLM failure) the provisional tokens are
dropped and only the ``error`` frame is emitted — the client never persists unguarded prose (§5).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator

from stock_agent.agent.events import AgentEvent, Chart, Error, Final, Sources, Tiles, Token
from stock_agent.agent.router import RouterError
from stock_agent.agent.runtime import AgentError
from stock_agent.ui.state import sources_from_tool_results
from stock_agent.ui.tiles import stat_tiles_from_tool_results
from stock_agent.viz.charts import charts_for


def adapt_events(events: Iterable[AgentEvent]) -> Iterator[AgentEvent]:
    """Weave adapter-owned ``tiles``/``chart``/``sources`` into the stream at the §5 positions.

    Passes head events (``turn_start``, ``route_decided``, ``tool_start``, ``tool_finish``) through
    live; buffers ``token``s; at ``Final`` emits ``tiles → chart* → token* → sources → final`` built
    from ``Final.tool_results``. Empty tiles/sources are omitted (matching the Streamlit builders
    and the mockup). ``Error`` discards buffered tokens and surfaces only the error.
    """
    buffered_tokens: list[Token] = []
    for ev in events:
        if isinstance(ev, Token):
            buffered_tokens.append(ev)  # provisional until Final (§5)
        elif isinstance(ev, Final):
            invs = ev.tool_results
            tiles = stat_tiles_from_tool_results(invs)
            if tiles:
                yield Tiles(tiles=tiles)
            for spec in charts_for(invs):
                yield Chart(spec=spec.to_dict())
            yield from buffered_tokens
            sources = sources_from_tool_results(invs)
            if sources:
                yield Sources(citations=sources)
            yield ev
        elif isinstance(ev, Error):
            yield ev  # discard the (provisional) buffered tokens — never persist unguarded prose
        else:
            yield ev  # turn_start / route_decided / tool_start / tool_finish stream live


def sse_frame(event: AgentEvent) -> str:
    """Serialize an event to an SSE frame ``data: <json>\\n\\n`` (the JSON carries its ``type``)."""
    return f"data: {json.dumps(event.to_wire())}\n\n"


def event_stream(events: Iterable[AgentEvent]) -> Iterator[str]:
    """Adapter → SSE frames, converting a terminal ``RouterError``/``AgentError`` into an ``error``.

    Grounding/max-iteration terminals already arrive as ``Error`` *events* (from
    ``run_agent_events``); the only exceptions that can escape are a ``RouterError`` (bad route /
    missing param, raised by ``run_events``) — surfaced here as a terminal ``error`` frame rather
    than a mid-stream 500, since the SSE response has already begun (200).
    """
    try:
        for ev in adapt_events(events):
            yield sse_frame(ev)
    except RouterError as exc:
        yield sse_frame(Error(code="router", message=str(exc)))
    except AgentError as exc:  # defensive: run_events surfaces these as events, never leaks a raise
        yield sse_frame(Error(code="agent", message=str(exc)))
