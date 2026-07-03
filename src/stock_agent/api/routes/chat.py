"""Streaming chat endpoint: ``POST /chat/stream`` (Phase 2, P2.2).

Thin — validate the request, ask the ``Router`` to run the turn as an ``AgentEvent`` stream, and
adapt those events to Server-Sent Events. No business logic, no numbers computed here (same rule as
``cli/``): tiles/chart/sources are built by the pure builders inside the adapter; the answer text
and tool results come from the runtime. Non-token for now (a single ``token`` carries the answer);
true token streaming lands in P2.4.
"""

from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from stock_agent.agent.router import Router
from stock_agent.api.deps import router_dep
from stock_agent.api.schemas import ChatStreamRequest
from stock_agent.api.streaming import event_stream

router = APIRouter()


@router.post("/chat/stream")
def post_chat_stream(
    req: ChatStreamRequest,
    router_obj: Annotated[Router, Depends(router_dep)],
) -> StreamingResponse:
    """Run one turn and stream its ``AgentEvent``s as SSE frames (ordering per plan §5).

    ``turn_id`` is client-supplied when correlating frames, else server-minted so ``turn_start`` /
    ``final`` always carry a stable id. ``X-Accel-Buffering: no`` disables proxy buffering so events
    flush incrementally (matters once P2.4 streams tokens).
    """
    turn_id = req.turn_id or uuid4().hex
    events = router_obj.run_events(
        req.message,
        route=req.route,
        ticker=req.ticker,
        horizon=req.horizon,
        days=req.days,
        model=req.model,
        history=req.history or None,
        thread_id=req.thread_id,
        turn_id=turn_id,
    )
    return StreamingResponse(
        event_stream(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
