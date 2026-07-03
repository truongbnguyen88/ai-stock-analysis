"""Chat agent (Role B): NL request -> tool calls -> grounded narration.

The agent routes; it never computes numbers. The numeric-grounding guard ensures
every figure it reports traces back to a tool result.
"""

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
from stock_agent.agent.guards import NumberGrounding
from stock_agent.agent.runtime import (
    AgentError,
    AgentGroundingError,
    AgentResult,
    AnthropicToolClient,
    ToolInvocation,
    ToolLLM,
    ToolResponse,
    ToolUse,
    run_agent,
    run_agent_events,
)
from stock_agent.agent.tools import TOOL_SCHEMAS, ToolExecutor

__all__ = [
    "NumberGrounding",
    "ToolExecutor",
    "TOOL_SCHEMAS",
    "run_agent",
    "run_agent_events",
    "AnthropicToolClient",
    "ToolInvocation",
    "ToolLLM",
    "ToolResponse",
    "ToolUse",
    "AgentResult",
    "AgentError",
    "AgentGroundingError",
    # Event schema (plan §4) — the SSE union yielded by run_agent_events / router / adapter.
    "AgentEvent",
    "TurnStart",
    "RouteDecided",
    "ToolStart",
    "ToolFinish",
    "Tiles",
    "Chart",
    "Token",
    "Sources",
    "Final",
    "Error",
    "hue_for",
]
