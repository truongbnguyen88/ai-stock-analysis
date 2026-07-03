"""SSE adapter tests (P2.2): §5 ordering, builder parity, empty-omission, error-discard.

Pure — feeds hand-built ``AgentEvent`` streams through ``adapt_events`` and asserts it weaves in
``tiles``/``chart``/``sources`` at the §5 positions *byte-identically* to the pure builders. No api,
no network. (The end-to-end SSE frame shape is covered by ``test_api_chat_stream``.)
"""

from __future__ import annotations

import json

from stock_agent.agent.events import (
    AgentEvent,
    Error,
    Final,
    RouteDecided,
    Tiles,
    Token,
    ToolFinish,
    ToolStart,
    TurnStart,
    hue_for,
)
from stock_agent.agent.runtime import ToolInvocation
from stock_agent.api.streaming import adapt_events, sse_frame
from stock_agent.ui.state import sources_from_tool_results
from stock_agent.ui.tiles import stat_tiles_from_tool_results

# A price result the tile builder recognizes (numbers straight from the tool).
_PRICE = {"last_close": 123.45, "pct_change": 0.05, "n_bars": 60}
# A filings-style result carrying citations (numbers/prose from the tool, cited).
_FILING = {"answer": "Revenue grew.", "citations": [{"marker": 1, "label": "NVDA 10-K"}]}


def _final(invs: list[ToolInvocation]) -> Final:
    return Final(
        tool_calls=[i.name for i in invs], iterations=1, messages=[], tool_results=invs,
        turn_id="tn",
    )


def test_adapter_weaves_tiles_and_sources_in_section5_order() -> None:
    invs = [
        ToolInvocation("get_price_summary", {}, _PRICE),  # → a tile, no chart
        ToolInvocation("search_filings", {}, _FILING),  # → a citation, no tile/chart
    ]
    raw: list[AgentEvent] = [
        TurnStart(thread_id="th", turn_id="tn", route="auto"),
        RouteDecided(mode="auto", route_name="auto"),
        ToolStart(tool="get_price_summary", input_summary="", hue_key=hue_for("get_price_summary")),
        ToolFinish(tool="get_price_summary", ok=True, elapsed_ms=1.0),
        ToolStart(tool="search_filings", input_summary="", hue_key=hue_for("search_filings")),
        ToolFinish(tool="search_filings", ok=True, elapsed_ms=1.0),
        Token(text="Here is the answer."),
        _final(invs),
    ]
    out = list(adapt_events(raw))
    assert [e.type for e in out] == [
        "turn_start", "route_decided",
        "tool_start", "tool_finish", "tool_start", "tool_finish",
        "tiles", "token", "sources", "final",  # tiles before token; sources after token; final last
    ]
    tiles = next(e for e in out if isinstance(e, Tiles))
    assert tiles.tiles == stat_tiles_from_tool_results(invs)  # byte-identical to the pure builder
    sources = next(e for e in out if e.type == "sources")
    assert sources.to_wire()["citations"] == sources_from_tool_results(invs)


def test_adapter_omits_empty_tiles_and_sources() -> None:
    # A turn whose tools produce neither tiles nor citations emits just the token + final.
    raw: list[AgentEvent] = [Token(text="No structured output."), _final([])]
    out = list(adapt_events(raw))
    assert [e.type for e in out] == ["token", "final"]


def test_adapter_error_discards_buffered_tokens() -> None:
    # Provisional tokens must never survive a terminal error (§5: client discards unguarded prose).
    raw: list[AgentEvent] = [
        Token(text="provisional 92.5%"),
        Error(code="grounding", message="ungrounded figure"),
    ]
    out = list(adapt_events(raw))
    assert [e.type for e in out] == ["error"]
    assert not any(isinstance(e, Token) for e in out)


def test_sse_frame_is_data_prefixed_json() -> None:
    frame = sse_frame(Token(text="hi"))
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    assert json.loads(frame[len("data: ") :].strip()) == {"type": "token", "text": "hi"}
