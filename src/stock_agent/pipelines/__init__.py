"""Use-case pipelines composing the core modules (thin orchestration)."""

from stock_agent.pipelines.analyze import run_analyze

__all__ = ["run_analyze"]
