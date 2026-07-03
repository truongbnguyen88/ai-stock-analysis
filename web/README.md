# web/ — React + Vite frontend (Phase 2)

Local-dev SPA for the stock-agent research console. Talks only to the FastAPI app in
`src/stock_agent/api/` over HTTP/SSE (plan: [docs/PHASE2_REACT_FASTAPI_PLAN.md](../docs/PHASE2_REACT_FASTAPI_PLAN.md)).

**Status:** P2.0 — static shell (top bar + sidebar from live `/config` + `/corpus`, instant
`data-theme` toggle). Conversation/streaming/export land in P2.1–P2.6.

## Run (two processes)

```bash
# 1) backend (repo root) — FastAPI on :8000
make api                       # == uvicorn stock_agent.api.app:app --reload --port 8000

# 2) frontend (this dir) — Vite dev on :5173
cd web && pnpm install         # first time (or: make web-install)
pnpm dev
```

Open http://localhost:5173. Point the SPA elsewhere with `VITE_API_BASE` (default `http://localhost:8000`).

## Checks

```bash
make check-web                 # token-bridge freshness + tsc --noEmit + vitest (from repo root)
# or, in web/:  pnpm typecheck && pnpm test
```

## Design tokens (do not hand-edit `src/tokens.css`)

`src/tokens.css` is **generated** from the Python design tokens in `src/stock_agent/ui/theme.py`
(single source of truth). Regenerate after any token change:

```bash
make gen-tokens                # == python scripts/gen_web_tokens.py
```

`tests/unit/test_token_bridge.py` (Python) and `src/test/tokens.test.ts` (web) fail if it drifts.
