"""Shared FastAPI dependencies for the api/ layer.

The settings dependency is indirected through this function so tests can override it via
``app.dependency_overrides[settings_dep]`` and never touch a real ``.env`` — the same
determinism rule the rest of the suite follows (no live config/network in tests).
"""

from __future__ import annotations

from stock_agent.settings import Settings, get_settings


def settings_dep() -> Settings:
    """FastAPI dependency yielding the app settings (overridable in tests)."""
    return get_settings()
