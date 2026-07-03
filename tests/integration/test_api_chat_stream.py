"""End-to-end ``POST /chat/stream`` tests (P2.2): SSE frame shape, §5 order, builder parity.

Deterministic: ``router_dep`` is overridden with a ``FakeToolLLM`` + ``FakeProvider``-backed
``Router`` (no network, no real Anthropic client). Asserts the client receives ``AgentEvent`` wire
frames in the §5 order and that ``tiles``/``chart`` are byte-identical to the pure builders — the
single source both frontends read, so React and Streamlit can't drift on numbers.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from fastapi.testclient import TestClient

from stock_agent.agent.events import Final
from stock_agent.agent.router import Router
from stock_agent.agent.runtime import ToolResponse
from stock_agent.agent.tools import ToolExecutor
from stock_agent.api.app import create_app
from stock_agent.api.deps import router_dep
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.settings import Settings
from stock_agent.ui.tiles import stat_tiles_from_tool_results
from stock_agent.viz.charts import charts_for


class FakeToolLLM:
    """Scripted ToolResponses in order (mirrors the runtime/router tests' fake)."""

    def __init__(self, responses: list[ToolResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, *, system, messages, tools, max_tokens) -> ToolResponse:  # type: ignore[no-untyped-def]
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


def _executor() -> ToolExecutor:
    bars = [
        PriceBar(
            date=date(2024, 1, 1) + timedelta(days=i),
            open=100.0 + i, high=100.0 + i + 0.5, low=100.0 + i - 0.5, close=100.0 + i,
        )
        for i in range(60)
    ]
    fake = FakeProvider("fake", prices=PriceSeries(ticker="NVDA", bars=bars))
    registry = ProviderRegistry([fake], Settings(_env_file=None, provider_price_priority="fake"))
    return ToolExecutor(Settings(_env_file=None), registry=registry)


def _final(text: str) -> ToolResponse:
    return ToolResponse(text=text, tool_uses=[], stop_reason="end_turn", assistant_content=[])


def _client(router: Router) -> TestClient:
    app = create_app()
    app.dependency_overrides[router_dep] = lambda: router
    return TestClient(app)


def _parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse an SSE body into the list of wire-dict payloads (one per ``data:`` frame)."""
    frames: list[dict[str, Any]] = []
    for block in body.strip().split("\n\n"):
        line = block.strip()
        if line.startswith("data:"):
            frames.append(json.loads(line[len("data:") :].strip()))
    return frames


def test_chat_stream_deterministic_order_and_builder_parity() -> None:
    router = Router(_executor())  # deterministic forecast route → no tool_llm needed
    resp = _client(router).post(
        "/chat/stream",
        json={
            "message": "forecast it", "route": "forecast", "ticker": "NVDA",
            "horizon": 20, "thread_id": "th", "turn_id": "tn",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse(resp.text)

    # §5 order — forecast yields tiles + a chart, no sources.
    assert [f["type"] for f in frames] == [
        "turn_start", "route_decided", "tool_start", "tool_finish",
        "tiles", "chart", "token", "final",
    ]
    assert frames[0]["thread_id"] == "th" and frames[0]["turn_id"] == "tn"
    assert frames[-1]["turn_id"] == "tn" and frames[-1]["grounded"] is True

    # Byte-identity to the pure builders (recompute from the same router's tool_results).
    invs = next(
        e for e in router.run_events("forecast it", route="forecast", ticker="NVDA", horizon=20)
        if isinstance(e, Final)
    ).tool_results
    tiles_frame = next(f for f in frames if f["type"] == "tiles")
    chart_frame = next(f for f in frames if f["type"] == "chart")
    assert tiles_frame["tiles"] == stat_tiles_from_tool_results(invs)
    assert chart_frame["spec"] == charts_for(invs)[0].to_dict()


def test_chat_stream_grounding_rejection_emits_error_not_final() -> None:
    # Persistent fabrication on the LLM path → terminal `error` (grounding), no `final`, no `token`.
    script = [_final("A 92.5% chance."), _final("Still 92.5% likely.")]
    router = Router(_executor(), tool_llm=FakeToolLLM(script))
    resp = _client(router).post("/chat/stream", json={"message": "will it moon", "route": "auto"})
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    types = [f["type"] for f in frames]

    assert "error" in types and "final" not in types and "token" not in types
    err = next(f for f in frames if f["type"] == "error")
    assert err["code"] == "grounding"


def test_chat_stream_bad_route_surfaces_error_frame() -> None:
    # A RouterError (missing ticker) becomes a terminal `error` frame, not a mid-stream 500.
    router = Router(_executor())
    resp = _client(router).post("/chat/stream", json={"message": "forecast", "route": "forecast"})
    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    err = next(f for f in frames if f["type"] == "error")
    assert err["code"] == "router" and "ticker" in err["message"]
