"""Chat agent (Role B): NL request -> tool calls -> grounded narration.

The agent routes; it never computes numbers. The numeric-grounding guard ensures
every figure it reports traces back to a tool result.
"""

from stock_agent.agent.guards import NumberGrounding
from stock_agent.agent.runtime import (
    AgentError,
    AgentGroundingError,
    AgentResult,
    AnthropicToolClient,
    ToolLLM,
    ToolResponse,
    ToolUse,
    run_agent,
)
from stock_agent.agent.tools import TOOL_SCHEMAS, ToolExecutor

__all__ = [
    "NumberGrounding",
    "ToolExecutor",
    "TOOL_SCHEMAS",
    "run_agent",
    "AnthropicToolClient",
    "ToolLLM",
    "ToolResponse",
    "ToolUse",
    "AgentResult",
    "AgentError",
    "AgentGroundingError",
]
