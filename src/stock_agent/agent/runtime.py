"""Agent runtime — the tool-use loop with numeric-grounding enforcement.

Drives a conversation: the model requests tools, we execute them, feed results
back, and repeat until it produces a final answer. Every number in tool results
is accumulated; the final answer is checked by the grounding guard. Ungrounded
figures trigger one corrective retry, then a refusal — the agent never knowingly
presents fabricated numbers.

``ToolLLM`` is the injection seam: ``AnthropicToolClient`` for production, a fake
for tests (so the suite never hits the network).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from stock_agent.agent.guards import NumberGrounding
from stock_agent.agent.prompts.agent import SYSTEM
from stock_agent.agent.tools import TOOL_SCHEMAS, ToolExecutor
from stock_agent.logging_config import get_logger
from stock_agent.settings import Settings

log = get_logger(__name__)

_MAX_ITERATIONS = 8
_MAX_TOKENS = 4096  # headroom for comprehensive multi-section answers


class AgentError(RuntimeError):
    """The agent could not produce a valid answer."""


class AgentGroundingError(AgentError):
    """The agent stated figures not supported by any tool result."""


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResponse:
    """Normalized model turn: any text, any tool requests, and the raw blocks."""

    text: str
    tool_uses: list[ToolUse]
    stop_reason: str
    assistant_content: Any  # raw content blocks, appended back to the transcript


@dataclass
class AgentResult:
    text: str
    tool_calls: list[str] = field(default_factory=list)
    iterations: int = 0


class ToolLLM(Protocol):
    """A tool-using chat model."""

    def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ToolResponse: ...


class AnthropicToolClient:
    """``ToolLLM`` backed by the Anthropic Messages API (tools + prompt caching)."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._model = settings.llm_model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic

            key = self._settings.require("anthropic_api_key", capability="agent chat")
            self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def create(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> ToolResponse:
        client = self._ensure_client()
        try:
            resp = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001 - normalize SDK/network errors
            raise AgentError(f"agent LLM request failed: {exc}") from exc

        text_parts: list[str] = []
        tool_uses: list[ToolUse] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
            elif getattr(block, "type", None) == "tool_use":
                tool_uses.append(ToolUse(id=block.id, name=block.name, input=dict(block.input)))
        return ToolResponse(
            text="".join(text_parts),
            tool_uses=tool_uses,
            stop_reason=resp.stop_reason,
            assistant_content=resp.content,
        )


def run_agent(
    query: str,
    *,
    llm: ToolLLM,
    executor: ToolExecutor,
    system: str = SYSTEM,
    tools: list[dict[str, Any]] | None = None,
    max_iterations: int = _MAX_ITERATIONS,
    grounding_retries: int = 1,
) -> AgentResult:
    """Run the tool-use loop and return a grounded final answer."""
    tools = tools if tools is not None else TOOL_SCHEMAS
    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
    grounding = NumberGrounding()
    tool_calls: list[str] = []
    retries_used = 0

    for iteration in range(1, max_iterations + 1):
        resp = llm.create(system=system, messages=messages, tools=tools, max_tokens=_MAX_TOKENS)

        if resp.tool_uses:
            messages.append({"role": "assistant", "content": resp.assistant_content})
            results: list[dict[str, Any]] = []
            for tu in resp.tool_uses:
                tool_calls.append(tu.name)
                result = executor.execute(tu.name, tu.input)
                grounding.add_from(result)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(result, default=str),
                    }
                )
            messages.append({"role": "user", "content": results})
            continue

        # Final answer: enforce numeric grounding.
        violations = grounding.ungrounded(resp.text)
        if violations and retries_used < grounding_retries:
            retries_used += 1
            log.warning("agent.grounding_retry", figures=violations)
            messages.append({"role": "assistant", "content": resp.assistant_content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "These figures are not supported by any tool result: "
                        f"{', '.join(violations)}. Restate using only values returned by tools "
                        "(call a tool if you need a number), or remove them."
                    ),
                }
            )
            continue
        if violations:
            raise AgentGroundingError(f"unverified figures after retry: {', '.join(violations)}")
        return AgentResult(text=resp.text, tool_calls=tool_calls, iterations=iteration)

    raise AgentError(f"agent did not finish within {max_iterations} iterations")
