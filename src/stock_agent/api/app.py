"""FastAPI application factory for the local-dev React frontend.

Local-dev posture only (PHASE2 plan §7/§11): CORS is opened to the Vite dev origin, no
auth, no HTTPS, no containers. Those are explicit non-goals until/unless the app is ever
deployed. Run with: ``uvicorn stock_agent.api.app:app --reload`` (or ``create_app()``).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stock_agent.api.routes import meta

# Vite dev server origins (default port 5173). Local-dev only; a real deploy would lock
# this down. Both host spellings are allowed so the browser's Origin header matches either.
DEV_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def create_app() -> FastAPI:
    """Build the FastAPI app: CORS for the Vite dev origin + the P2.0 meta routes."""
    app = FastAPI(
        title="stock-agent api",
        version="0.0.1",
        summary="Local-dev API for the React frontend (research/education only; not advice).",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(meta.router)
    return app


# Module-level app for ``uvicorn stock_agent.api.app:app``.
app = create_app()
