"""FastAPI app for the Phase-2 React frontend (docs/PHASE2_REACT_FASTAPI_PLAN.md).

New top-level entrypoint, analogous to ``cli/``: it validates requests, calls the
existing pure contracts (``corpus_status``, routing/settings metadata; streaming runtime
lands in later slices), and returns JSON/SSE. **Depends downward only** — nothing in the
core package imports ``api``. No business logic lives here (same rule as ``cli/``).

P2.0 surface: ``GET /config`` and ``GET /corpus`` only (no agent calls yet).
"""

from __future__ import annotations

from stock_agent.api.app import create_app

__all__ = ["create_app"]
