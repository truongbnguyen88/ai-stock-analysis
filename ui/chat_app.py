"""Streamlit chat frontend for the stock research agent — orchestration only (R1).

Launch with:
    streamlit run ui/chat_app.py

Thin by design: page config, shared resources, session init, then wire the view
components (sidebar / hero / message) around the turn loop. Rendering lives in
``ui/components/``, session/persistence in ``ui/session.py``, pure transforms in
``stock_agent.ui.*``. Requires ANTHROPIC_API_KEY (and optionally news provider keys)
in .env. This is NOT financial advice.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing in editable mode: the package
# lives under src/, and the sibling view modules (session, components) sit next to this
# script — add both so `streamlit run ui/chat_app.py` resolves them.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ---- page config (must be the first Streamlit call) ----
st.set_page_config(
    page_title="Stock Research Agent",
    page_icon="📈",
    layout="wide",
)

import session
from components.hero import render_capability_hero
from components.message import (
    render_chart,
    render_export,
    render_sources,
    render_stat_tiles,
    render_trace,
)
from components.sidebar import render_sidebar
from stock_agent.agent.router import Router, RouterError, resolve_domain
from stock_agent.agent.runtime import (
    AgentError,
    AgentGroundingError,
    AnthropicToolClient,
    ToolInvocation,
    run_agent,
)
from stock_agent.agent.tools import ToolExecutor
from stock_agent.chat.history import ChatStore
from stock_agent.llm.client import AnthropicClient
from stock_agent.settings import get_settings
from stock_agent.ui.routing import chat_input_placeholder
from stock_agent.ui.state import sources_from_tool_results
from stock_agent.ui.tiles import stat_tiles_from_tool_results
from stock_agent.ui.theme import theme_style_tag
from stock_agent.viz.charts import ChartSpec, charts_for

# ---- design system: inject the brass-on-ink token block + base typography once (R0). ----
# Pure CSS from the theme module (no Streamlit dependency there); Streamlit's native
# chrome colors come from .streamlit/config.toml. See docs/APP_REDESIGN.md §5.
st.markdown(theme_style_tag(), unsafe_allow_html=True)


# ---- shared (cached) agent resources — one set per Streamlit session ----
@st.cache_resource
def load_resources():
    """Build shared agent resources; open the chat store and prune expired threads once."""
    settings = get_settings()
    executor = ToolExecutor(settings, llm=AnthropicClient(settings))
    llm = AnthropicToolClient(settings)
    store = ChatStore(
        settings.chat_history_dir, retention_days=settings.chat_history_retention_days
    )
    store.prune()
    return settings, executor, llm, store


settings, executor, agent_llm, store = load_resources()
# Deterministic-routing front door (the domain selector dispatches through this; no
# tool_llm needed because the deterministic path never makes a routing LLM call). Auto
# mode still uses run_agent.
router = Router(executor)

# ---- session state + sidebar (which lists/opens/deletes threads) ----
session.init_session_state()
choice = render_sidebar(settings=settings, store=store)

# ---- header ----
st.title("Stock Research Agent")
st.caption(
    "Ask about any ticker — the agent calls tools for prices, indicators, news, forecasts, "
    "and SEC filings (single-shot or multi-hop)."
)

# Empty-state hero: discoverable capability showcase, shown only on a fresh chat.
if not st.session_state.messages:
    render_capability_hero(choice.ticker)

# ---- render history ----
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        # Stat tiles first (summary-before-detail); persisted with the turn so they
        # re-render after a rerun/restart, like charts (R4). The tool-trace chip row is
        # intentionally live-turn-only, so it is not re-rendered here.
        if msg["role"] == "assistant":
            render_stat_tiles(msg.get("tiles", []))
        st.markdown(msg["content"])
        # Re-render any charts captured with this turn (survives Streamlit reruns).
        for spec in msg.get("charts", []):
            render_chart(spec)
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            render_export(msg["content"], msg.get("charts", []), idx)

# ---- handle quick-starter injection ----
pending = st.session_state.pop("pending_prompt", None)
# When a deterministic route can't handle the input, the user can bounce the SAME text to
# the LLM router via an inline button. That sets `force_auto_prompt`, which overrides the
# sidebar routing mode for this one turn only.
force_auto_prompt = st.session_state.pop("force_auto_prompt", None)

# ---- input bar ----
if not settings.anthropic_api_key:
    st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

# Placeholder hints what to type for the active routing mode.
_placeholder = chat_input_placeholder(choice)

prompt = st.chat_input(_placeholder) or pending or force_auto_prompt
# A forced-Auto retry runs the LLM router regardless of the sidebar mode; otherwise the
# sidebar mode (Auto vs. a named domain) decides.
run_auto = choice.is_auto or (
    force_auto_prompt is not None and prompt == force_auto_prompt
)

if prompt:
    # Starting a fresh turn clears any pending bounce-to-Auto offer from a failed turn.
    st.session_state.pop("fallback_prompt", None)
    # Show user message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the request and render the response.
    with st.chat_message("assistant"):
        charts: list[ChartSpec] = []
        sources: list[dict] = []
        tiles: list[dict] = []
        tool_names: list[str] = []  # tools that ran (drives the live trace chip row)
        if run_auto:
            # ---- Auto: the LLM agent loop picks + composes tools. ----
            # st.status gives a step-list surface (approximates a live per-tool trace;
            # run_agent is a single blocking call, so it resolves to a done/error summary).
            with st.status("Working on your request…", expanded=False) as status:
                try:
                    result = run_agent(
                        prompt,
                        llm=agent_llm,
                        executor=executor,
                        history=st.session_state.agent_history,  # pass prior turns
                    )
                    response = result.text
                    # Persist the full message history for the next turn.
                    st.session_state.agent_history = result.messages
                    # Charts + tiles derived from the tool results (numbers the tools
                    # produced, never the LLM) — rendered alongside the text below.
                    charts = charts_for(result.tool_results)
                    tiles = stat_tiles_from_tool_results(result.tool_results)
                    # Resolved SEC citations from the RAG tools (from tool output, not LLM).
                    sources = sources_from_tool_results(result.tool_results)
                    tool_names = (
                        list(dict.fromkeys(result.tool_calls))  # preserve order, dedup
                        if result.tool_calls
                        else []
                    )
                    status.update(
                        label=(
                            f"Done · {len(tool_names)} tool(s)" if tool_names else "Done"
                        ),
                        state="complete",
                    )
                except AgentGroundingError as exc:
                    response = (
                        f"⚠️ The agent produced unverifiable figures and was stopped.\n\n"
                        f"Details: {exc}\n\n"
                        "Please rephrase your question or ask the agent to call a specific tool."
                    )
                    status.update(label="Stopped (unverifiable figures)", state="error")
                except AgentError as exc:
                    low = str(exc).lower()
                    if "tim" in low or "connection" in low or "interrupt" in low:
                        response = (
                            "⚠️ The request timed out (often a temporary network hiccup or heavy "
                            "background load). Please try again — your conversation is preserved."
                        )
                    else:
                        response = f"⚠️ Agent error: {exc}"
                    status.update(label="Agent error", state="error")
        else:
            # ---- Deterministic: dispatch the chosen domain (no routing LLM call). ----
            # A deterministic turn is standalone (not part of the LLM dialogue), so
            # agent_history is left untouched. We wrap the single tool result in a
            # ToolInvocation to reuse the SAME chart + tile + citation rendering as Auto.
            with st.spinner(f"Running {choice.domain}…"):
                try:
                    route = resolve_domain(choice.domain, choice.variant)
                    rr = router.run(
                        prompt,
                        route=route,
                        ticker=choice.ticker,
                        horizon=choice.horizon,
                        days=choice.days,
                    )
                    structured = rr.structured or {}
                    tool_name = rr.tool_calls[0] if rr.tool_calls else route
                    invs = [ToolInvocation(name=tool_name, input={}, result=structured)]
                    charts = charts_for(invs)
                    tiles = stat_tiles_from_tool_results(invs)
                    sources = sources_from_tool_results(invs)
                    tool_names = [tool_name]  # the single deterministic route
                    if "error" in structured:
                        response = f"⚠️ {structured['error']}"
                        # LLM router may handle what this route couldn't — offer the bounce.
                        st.session_state["fallback_prompt"] = prompt
                    elif "answer" in structured:
                        response = rr.text  # synthesis tools carry a cited prose answer
                    else:
                        # Numeric tools: show the structured result verbatim (charts below).
                        response = (
                            f"**{choice.domain} · {choice.variant}** "
                            f"(`{tool_name}`)\n\n```json\n{rr.text}\n```"
                        )
                except RouterError as exc:
                    response = f"⚠️ {exc}"
                    # e.g. a missing ticker/param the LLM router can infer from the text.
                    st.session_state["fallback_prompt"] = prompt
        # ---- render: tiles first (summary), then answer, chart, trace, sources, export ----
        render_stat_tiles(tiles)
        st.markdown(response)
        for spec in charts:
            render_chart(spec)
        render_trace(tool_names, is_auto=run_auto)
        render_sources(sources)
        # Export buttons for this answer (idx = the index it gets once appended below).
        render_export(response, charts, len(st.session_state.messages))

    # Persist text + tiles + charts + filing sources so the turn re-renders intact on reruns.
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "charts": charts,
            "sources": sources,
            "tiles": tiles,
        }
    )
    # Save the thread to disk so it survives restarts and shows in the sidebar.
    session.save_current_thread(store)

# ---- bounce-to-Auto affordance (OUTSIDE the turn block so the click is captured) ----
# A Streamlit button must be re-instantiated on the run that handles its click; a button
# defined only inside `if prompt:` would be gone on the click rerun (prompt is empty then),
# so its click would be lost. We persist the failed text in `fallback_prompt` and render
# the button here every run until the user acts. Explicit (not automatic): a deterministic
# turn never silently spends a routing LLM call the user opted out of by choosing a domain.
_fallback = st.session_state.get("fallback_prompt")
if _fallback and st.button(
    "🤖 Ask this in Auto (LLM) mode instead",
    key="auto_fallback_btn",
    help="Re-run this exact question through the LLM router, which can compose multiple tools.",
):
    st.session_state["force_auto_prompt"] = _fallback
    st.session_state.pop("fallback_prompt", None)
    st.rerun()
