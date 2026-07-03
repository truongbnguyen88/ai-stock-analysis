"""Shared FastAPI dependencies for the api/ layer.

The settings dependency is indirected through this function so tests can override it via
``app.dependency_overrides[settings_dep]`` and never touch a real ``.env`` — the same
determinism rule the rest of the suite follows (no live config/network in tests). ``router_dep``
is likewise overridden in tests with a ``FakeToolLLM``/``FakeProvider``-backed ``Router`` (no
network, no real Anthropic client).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends

from stock_agent.settings import Settings, get_settings

if TYPE_CHECKING:  # keep the meta path (settings_dep) free of the agent import at module load
    from stock_agent.agent.router import Router


def settings_dep() -> Settings:
    """FastAPI dependency yielding the app settings (overridable in tests)."""
    return get_settings()


def router_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> Router:
    """Build the hybrid ``Router`` for ``/chat/stream`` (deterministic + LLM paths).

    The executor's own LLM powers synthesis tools (news/filings); the agent ``tool_llm`` powers LLM
    routing. Both are ``None`` without an Anthropic key — numeric deterministic routes still work;
    an LLM-path request then fails loudly with a ``RouterError`` (surfaced as an ``error`` frame).
    Imports are local so non-api code paths don't pay the agent import cost. Returns ``object`` to
    keep the meta path free of the agent import; the type is ``Router`` (checked, not loaded here).
    """
    from stock_agent.agent.router import Router
    from stock_agent.agent.runtime import AnthropicToolClient
    from stock_agent.agent.tools import ToolExecutor
    from stock_agent.llm.client import AnthropicClient

    has_key = bool(settings.anthropic_api_key)
    executor = ToolExecutor(settings, llm=AnthropicClient(settings) if has_key else None)
    tool_llm = AnthropicToolClient(settings) if has_key else None
    return Router(executor, tool_llm=tool_llm)
