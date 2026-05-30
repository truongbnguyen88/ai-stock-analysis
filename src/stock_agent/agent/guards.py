"""Numeric-grounding guard for the chat agent (Role B).

The guard implementation now lives in ``llm.guards`` (shared with the synthesizer,
Role C) — re-exported here for the agent's use.
"""

from __future__ import annotations

from stock_agent.llm.guards import NumberGrounding

__all__ = ["NumberGrounding"]
