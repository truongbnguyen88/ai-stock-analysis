"""AgentEvent schema tests (P2.1): wire serialization shape + hue determinism.

These pin the SSE frame contract (plan §4) so a schema change breaks loudly. No runtime,
no network — pure dataclass/serialization checks.
"""

from __future__ import annotations

import json

from stock_agent.agent.events import (
    AgentEvent,
    Chart,
    Error,
    Final,
    RouteDecided,
    Sources,
    Tiles,
    Token,
    ToolFinish,
    ToolStart,
    TurnStart,
    hue_for,
)


def test_hue_for_is_deterministic_and_name_dependent() -> None:
    # crc32-based → stable across calls/processes (unlike salted hash()), and distinct tools differ.
    assert hue_for("run_forecast") == hue_for("run_forecast")
    assert hue_for("run_forecast") != hue_for("get_news_sentiment")
    assert isinstance(hue_for("compute_indicators"), int)


def test_to_wire_shapes_match_the_p2_4_schema() -> None:
    # Each event serializes to exactly the documented §4 payload keys (plus the `type` tag).
    assert TurnStart(thread_id="t", turn_id="u", route="auto", ticker="NVDA").to_wire() == {
        "type": "turn_start",
        "thread_id": "t",
        "turn_id": "u",
        "route": "auto",
        "ticker": "NVDA",
    }
    assert RouteDecided(mode="deterministic", route_name="forecast", note="n").to_wire() == {
        "type": "route_decided",
        "mode": "deterministic",
        "route_name": "forecast",
        "note": "n",
    }
    assert ToolStart(tool="run_forecast", input_summary="ticker=NVDA", hue_key=7).to_wire() == {
        "type": "tool_start",
        "tool": "run_forecast",
        "input_summary": "ticker=NVDA",
        "hue_key": 7,
    }
    assert ToolFinish(tool="run_forecast", ok=True, elapsed_ms=12.345).to_wire() == {
        "type": "tool_finish",
        "tool": "run_forecast",
        "ok": True,
        "elapsed_ms": 12.3,  # rounded to 0.1 ms for the wire
    }
    assert Tiles(tiles=[{"label": "Vol", "value": "58%"}]).to_wire() == {
        "type": "tiles",
        "tiles": [{"label": "Vol", "value": "58%"}],
    }
    assert Chart(spec={"kind": "bar"}).to_wire() == {"type": "chart", "spec": {"kind": "bar"}}
    assert Token(text="hello").to_wire() == {"type": "token", "text": "hello"}
    assert Sources(citations=[{"marker": 1, "label": "10-K"}]).to_wire() == {
        "type": "sources",
        "citations": [{"marker": 1, "label": "10-K"}],
    }
    assert Error(code="grounding", message="bad").to_wire() == {
        "type": "error",
        "code": "grounding",
        "message": "bad",
    }


def test_final_wire_omits_server_only_fields() -> None:
    # Final carries messages/tool_results IN-PROCESS (for the drain) but the wire frame is the §4
    # subset — the answer text already streamed as `token`s; the transcript stays server-side.
    final = Final(
        tool_calls=["run_forecast"],
        iterations=2,
        messages=[{"role": "assistant", "content": "x"}],
        tool_results=[],  # ToolInvocation list; not serialized to the wire
        turn_id="u1",
    )
    wire = final.to_wire()
    assert wire == {
        "type": "final",
        "turn_id": "u1",
        "tool_calls": ["run_forecast"],
        "iterations": 2,
        "grounded": True,
    }
    assert "messages" not in wire and "tool_results" not in wire and "text" not in wire


def test_every_wire_frame_is_json_serializable() -> None:
    # The adapter will json.dumps each frame; guard against a non-serializable payload sneaking in.
    events: list[AgentEvent] = [
        TurnStart(thread_id="t", turn_id="u", route="auto"),
        RouteDecided(mode="auto", route_name="agent"),
        ToolStart(tool="x", input_summary="", hue_key=hue_for("x")),
        ToolFinish(tool="x", ok=False, elapsed_ms=1.0),
        Tiles(tiles=[]),
        Chart(spec={}),
        Token(text="t"),
        Sources(citations=[]),
        Final(tool_calls=[], iterations=1, messages=[], tool_results=[]),
        Error(code="agent", message="m"),
    ]
    for ev in events:
        json.dumps(ev.to_wire())  # raises on failure
