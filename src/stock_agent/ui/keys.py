"""API-key availability for the sidebar KEYS row — one shared source of truth.

Both frontends surface the same set: the Streamlit sidebar (``ui/components/sidebar.py``,
via ``ui.html.keys_row``) and the React app (``api/routes/meta.py`` → ``GET /config`` →
``KeyStatus``). Deriving the set here — in the pure, gate-tested ``ui`` layer — means the two
can't drift (the same rule the token bridge follows).

State semantics (consumed by ``ui.html.keys_row`` / the React chip): present → ✓; missing &
required → ✕; missing & optional → ·. **Only Anthropic is a hard app-level requirement** (the
LLM); everything else is optional at the app level (the app runs without it — news has keyless
fallbacks; local embeddings need no key) and is surfaced so the user can see what is configured.
"""

from __future__ import annotations

from stock_agent.settings import Settings


def key_statuses(settings: Settings) -> list[tuple[str, bool, bool]]:
    """Return ``(label, present, required)`` rows for the KEYS chip row, in display order.

    - Anthropic (required) + the three keyed default news providers (optional; keyless
      Google News RSS / GDELT cover the zero-key case).
    - ``SEC EDGAR`` (``sec_user_agent``) and ``Voyage`` gate the SEC-filings / RAG capabilities
      the app advertises. Both are marked **optional** — needed only for that path, not to run the
      app. The Voyage row appears **only when Voyage is the active embedder**
      (``embedding_provider == "voyage"``), so ``local`` / ``openai`` users don't see an irrelevant
      key. (Local embeddings need no key; the SEC UA is only used to download filings.)
    """
    rows: list[tuple[str, bool, bool]] = [
        ("Anthropic", bool(settings.anthropic_api_key), True),
        ("Finnhub", bool(settings.finnhub_api_key), False),
        ("Marketaux", bool(settings.marketaux_api_key), False),
        ("Alpha Vantage", bool(settings.alpha_vantage_api_key), False),
        ("SEC EDGAR", bool(settings.sec_user_agent), False),
    ]
    if settings.embedding_provider == "voyage":
        rows.append(("Voyage", bool(settings.voyage_api_key), False))
    return rows
