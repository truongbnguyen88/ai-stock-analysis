"""Sidebar: brand, corpus status, quick starters, routing, chats, keys (redesign R1).

Extracted verbatim from ``ui/chat_app.py`` — behavior-preserving. Renders the whole
sidebar and returns the resolved :class:`RoutingChoice` for the turn handler + input bar.
Thread actions (new/open/delete) delegate to ``ui.session``.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

import session
from stock_agent.agent.router import DOMAIN_NAMES, DOMAINS
from stock_agent.chat.history import ChatStore
from stock_agent.rag.status import corpus_status
from stock_agent.settings import Settings, get_settings
from stock_agent.ui.routing import AUTO_MODE, RoutingChoice

_QUICK_STARTERS = {
    "📰 News summary": "Summarize the latest news for {ticker} and identify the key themes, risks, and catalysts.",
    "📊 Technical analysis": "Give me a technical analysis of {ticker}: trend, momentum, and volatility.",
    "🔮 Forecast 20 days": "What does the historical model forecast for {ticker} over the next 20 trading days?",
    "🔍 Full analysis": "Analyze {ticker} over the last 30 days — technicals, news summary, and 20-day forecast.",
}


@st.cache_data(ttl=300, show_spinner=False)
def _corpus_status() -> dict[str, Any]:
    """Cached corpus snapshot (active embedder + chunk count + freshness) for the badge."""
    return corpus_status(get_settings()).model_dump()


def render_sidebar(*, settings: Settings, store: ChatStore) -> RoutingChoice:
    """Render the sidebar and return the routing selection for this turn."""
    with st.sidebar:
        st.title("📈 Stock Research Agent")
        st.caption("Research / education only — **not financial advice**.")
        # Which embeddings filing questions are answered from (e.g. voyage-4) + freshness.
        _cs = _corpus_status()
        _chunks = "unavailable" if _cs["chunks"] < 0 else f"{_cs['chunks']:,} chunks"
        st.caption(f"📚 **Filing search:** `{_cs['embedder']}` — {_chunks}")
        if _cs["latest"]:
            st.caption(f"{_cs['tickers']} tickers · fresh to {_cs['latest']}")
        st.divider()

        st.subheader("Quick starters")
        ticker_input = st.text_input("Ticker", value="NVDA", max_chars=10).upper().strip()

        for label, template in _QUICK_STARTERS.items():
            if st.button(label, use_container_width=True):
                st.session_state.pending_prompt = template.replace("{ticker}", ticker_input)

        st.divider()
        # ---- Hybrid routing: Auto (LLM picks tools) vs. a deterministic domain ----
        st.subheader("Routing")
        route_mode = st.selectbox(
            "Mode",
            [AUTO_MODE, *DOMAIN_NAMES],
            help=(
                "Auto lets the LLM choose and compose tools. A domain runs that one capability "
                "directly — no routing LLM call (faster/cheaper when you know what you want)."
            ),
        )
        selected_domain: str | None = None if route_mode == AUTO_MODE else route_mode
        selected_variant: str | None = None
        horizon_val: int | None = None
        days_val: int | None = None
        if selected_domain is not None:
            dom = DOMAINS[selected_domain]
            st.caption(dom.blurb)
            variant_labels = list(dom.variants)
            selected_variant = st.selectbox(
                "Variant", variant_labels, index=variant_labels.index(dom.default)
            )
            # Only surface the params the chosen domain actually uses (0 = use tool default).
            if selected_domain == "predictions":
                _h = st.number_input("Horizon (days)", min_value=0, value=20, step=5)
                horizon_val = int(_h) or None
            if selected_domain in ("news", "technicals", "brief"):
                _d = st.number_input("Lookback (days)", min_value=0, value=14, step=1)
                days_val = int(_d) or None
            _target = dom.variants[selected_variant]
            st.caption(f"🎯 Deterministic — runs `{_target}`, skips LLM routing.")

        st.divider()
        st.subheader("💬 Chats")
        if st.button("➕ New chat", use_container_width=True):
            session.save_current_thread(store)  # keep the one we're leaving
            session.start_new_thread()
            st.rerun()

        for meta in store.list_threads():
            is_current = meta.id == st.session_state.thread_id
            row, trash = st.columns([0.82, 0.18])
            label = ("🟢 " if is_current else "") + meta.title
            if row.button(label, key=f"open_{meta.id}", use_container_width=True):
                session.save_current_thread(store)
                session.open_thread(store, meta.id)
                st.rerun()
            if trash.button("🗑️", key=f"del_{meta.id}", help="Delete this chat"):
                store.delete(meta.id)
                if is_current:
                    session.start_new_thread()
                st.rerun()
        st.caption(f"Saved for {settings.chat_history_retention_days} days.")

        st.divider()
        st.caption(
            "**Keys configured:**\n"
            + ("✅ Anthropic\n" if settings.anthropic_api_key else "❌ Anthropic (required)\n")
            + ("✅ Finnhub\n" if settings.finnhub_api_key else "⬜ Finnhub\n")
            + ("✅ Marketaux\n" if settings.marketaux_api_key else "⬜ Marketaux\n")
            + ("✅ Alpha Vantage" if settings.alpha_vantage_api_key else "⬜ Alpha Vantage")
        )

    return RoutingChoice(
        ticker=ticker_input,
        domain=selected_domain,
        variant=selected_variant,
        horizon=horizon_val,
        days=days_val,
    )
