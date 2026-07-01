"""Streamlit chat frontend for the stock research agent.

Launch with:
    streamlit run ui/chat_app.py

Requires ANTHROPIC_API_KEY (and optionally news provider keys) in .env.
This is NOT financial advice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the repo root without installing in editable mode.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import streamlit.components.v1 as components

# ---- page config (must be the first Streamlit call) ----
st.set_page_config(
    page_title="Stock Research Agent",
    page_icon="📈",
    layout="wide",
)

from stock_agent.agent.router import (
    DOMAIN_NAMES,
    DOMAINS,
    Router,
    RouterError,
    resolve_domain,
)
from stock_agent.agent.runtime import (
    AgentError,
    AgentGroundingError,
    AnthropicToolClient,
    ToolInvocation,
    run_agent,
)
from stock_agent.agent.tools import ToolExecutor
from stock_agent.chat.history import (
    ChatStore,
    ChatThread,
    derive_title,
    new_thread_id,
    now_iso,
)
from stock_agent.llm.client import AnthropicClient
from stock_agent.rag.status import corpus_status
from stock_agent.reports.export import EXPORT_META, export_summary
from stock_agent.settings import get_settings
from stock_agent.ui.capabilities import CAPABILITIES
from stock_agent.viz.charts import ChartSpec, charts_for
from stock_agent.viz.render import to_altair, to_png


def _render_chart(spec: ChartSpec) -> None:
    """Render one ChartSpec as an Altair chart alongside the agent's text."""
    st.altair_chart(to_altair(spec).properties(height=260), use_container_width=True)
    if spec.caption:
        st.caption(spec.caption)


@st.cache_data(show_spinner=False)
def _chart_pngs(charts_json: str) -> list[bytes]:
    """Render a turn's charts to PNG bytes (cached per chart set; vl-convert)."""
    if not charts_json or charts_json == "[]":
        return []
    specs = [ChartSpec.from_dict(d) for d in json.loads(charts_json)]
    return [to_png(s) for s in specs]


@st.cache_data(show_spinner=False)
def _export_bytes(text: str, fmt: str, charts_json: str) -> bytes:
    """Render an answer (text + chart figures) to document bytes; cached across reruns."""
    images = _chart_pngs(charts_json) if fmt in ("pdf", "docx") else None
    return export_summary(text, fmt, images=images)


def _render_export(text: str, charts: list[ChartSpec], idx: int) -> None:
    """PDF / Word / Markdown download buttons for an assistant answer (with figures)."""
    charts_json = json.dumps([c.to_dict() for c in charts])
    with st.expander("📄 Export this summary"):
        cols = st.columns(3)
        labels = {"pdf": "PDF", "docx": "Word", "md": "Markdown"}
        for col, fmt in zip(cols, ["pdf", "docx", "md"]):
            mime, ext = EXPORT_META[fmt]
            col.download_button(
                labels[fmt],
                data=_export_bytes(text, fmt, charts_json),
                file_name=f"stock_summary.{ext}",
                mime=mime,
                key=f"dl_{idx}_{fmt}",
                use_container_width=True,
            )

def _sources_from_tool_results(tool_results: list) -> list[dict]:  # type: ignore[type-arg]
    """Collect SEC citations from the RAG tools (search_filings / research_multistep /
    research_summary).

    Tool-name agnostic: pulls any ``citations`` field from any tool result, so new RAG tools
    surface automatically. Citations come from the tool OUTPUT (P7/P8 resolved them against the
    retrieved set), not the LLM — so surfacing them here can't introduce a fabricated source.
    Deduped by (marker, label), preserving first-seen order.
    """
    seen: set[tuple[int, str]] = set()
    sources: list[dict] = []  # type: ignore[type-arg]
    for inv in tool_results:
        for cit in inv.result.get("citations", []) if isinstance(inv.result, dict) else []:
            key = (cit.get("marker"), cit.get("label", ""))
            if key not in seen:
                seen.add(key)
                sources.append({"marker": cit.get("marker"), "label": cit.get("label", "")})
    return sources


def _render_sources(sources: list[dict]) -> None:  # type: ignore[type-arg]
    """Render a turn's resolved SEC citations as a compact 'Filing sources' expander."""
    if not sources:
        return
    with st.expander(f"📑 Filing sources ({len(sources)})"):
        for s in sources:
            st.caption(f"[{s['marker']}] {s['label']}")


def _render_typewriter() -> None:
    """Animated centerpiece: a pure client-side typewriter cycling capability headlines.

    Rendered in an iframe via ``components.html`` so it never triggers a server
    rerun or blocks the chat. The iframe does NOT inherit Streamlit's theme, so
    colors adapt via ``prefers-color-scheme``; motion honors
    ``prefers-reduced-motion`` (static first line). Fixed height reserves space so
    typed text never clips. The cards below are the actionable layer — this is the
    visual only (A.3).
    """
    phrases = [f"{cap.title}  {cap.icon}" for cap in CAPABILITIES]
    html = """
<div class="tw-wrap">
  <span class="tw-prefix">You can ask about&nbsp;</span><span id="tw-text"></span><span class="tw-caret">▌</span>
</div>
<style>
  .tw-wrap {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 1.5rem; font-weight: 600; text-align: center;
    height: 2.4rem; line-height: 2.4rem; color: #1f2937;
  }
  .tw-prefix { opacity: 0.6; }
  .tw-caret { animation: tw-blink 1s step-end infinite; opacity: 0.7; }
  @keyframes tw-blink { 50% { opacity: 0; } }
  @media (prefers-color-scheme: dark) { .tw-wrap { color: #e5e7eb; } }
  @media (prefers-reduced-motion: reduce) { .tw-caret { animation: none; } }
</style>
<script>
  const phrases = __PHRASES__;
  const el = document.getElementById("tw-text");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce) {
    el.textContent = phrases[0];  // static, no animation
  } else {
    let pi = 0, ci = 0, deleting = false;
    function tick() {
      const full = phrases[pi];
      ci += deleting ? -1 : 1;
      el.textContent = full.slice(0, ci);
      let delay = deleting ? 35 : 70;
      if (!deleting && ci === full.length) { delay = 1500; deleting = true; }
      else if (deleting && ci === 0) { deleting = false; pi = (pi + 1) % phrases.length; delay = 300; }
      setTimeout(tick, delay);
    }
    tick();
  }
</script>
"""
    html = html.replace("__PHRASES__", json.dumps(phrases))
    components.html(html, height=58)


def _render_capability_hero(ticker: str) -> None:
    """Empty-state showcase: animated typewriter (A.3) over a grid of capability cards.

    Data-driven from ``CAPABILITIES`` (single source of truth) — the typewriter is
    the visual, the cards are the action. Clicking a card reuses the existing
    quick-starter plumbing — set ``pending_prompt`` + rerun, which the
    ``st.chat_input(...) or pending`` line below submits.
    """
    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        _render_typewriter()
        st.markdown(
            "#### What can this agent do?\n"
            "Pick a capability to try it on **"
            f"{ticker or 'a ticker'}** — or just ask in the box below.",
        )
        n_cols = 2
        for row_start in range(0, len(CAPABILITIES), n_cols):
            cols = st.columns(n_cols)
            for col, cap in zip(cols, CAPABILITIES[row_start : row_start + n_cols]):
                with col:
                    if st.button(
                        f"{cap.icon}  **{cap.title}**\n\n{cap.blurb}",
                        key=f"cap_{row_start}_{cap.title}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_prompt = cap.render_example(ticker)
                        st.rerun()


# ---- load settings once ----
@st.cache_resource
def load_resources():
    """Build shared (cached) agent resources — one set per Streamlit session.

    Also opens the chat-history store and prunes expired threads once per session.
    """
    settings = get_settings()
    executor = ToolExecutor(settings, llm=AnthropicClient(settings))
    llm = AnthropicToolClient(settings)
    store = ChatStore(
        settings.chat_history_dir, retention_days=settings.chat_history_retention_days
    )
    store.prune()
    return settings, executor, llm, store

settings, executor, agent_llm, store = load_resources()
# Deterministic-routing front door (the domain selector below dispatches through this; no tool_llm
# needed because the deterministic path never makes a routing LLM call). Auto mode still uses run_agent.
router = Router(executor)
_AUTO_MODE = "🤖 Auto (LLM router)"


# ---- chat-thread persistence helpers ----
def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Display messages with charts flattened to dicts (for the store)."""
    out = []
    for m in messages:
        sm = {"role": m["role"], "content": m["content"]}
        if m.get("charts"):
            sm["charts"] = [c.to_dict() for c in m["charts"]]
        if m.get("sources"):
            sm["sources"] = m["sources"]  # plain dicts already; survive a restart
        out.append(sm)
    return out


def _deserialize_messages(raw: list[dict]) -> list[dict]:
    """Inverse of _serialize_messages — rebuild ChartSpecs for rendering."""
    out = []
    for m in raw:
        out.append(
            {
                "role": m["role"],
                "content": m["content"],
                "charts": [ChartSpec.from_dict(c) for c in m.get("charts", [])],
                "sources": m.get("sources", []),
            }
        )
    return out


def _save_current_thread() -> None:
    """Persist the active conversation (skips an empty/unstarted thread)."""
    msgs = st.session_state.messages
    if not msgs:
        return
    store.save(
        ChatThread(
            id=st.session_state.thread_id,
            title=derive_title(msgs),
            created_at=st.session_state.thread_created_at,
            updated_at=now_iso(),
            display_messages=_serialize_messages(msgs),
            agent_history=st.session_state.agent_history,
        )
    )


def _clear_transient_ui_state() -> None:
    """Drop per-turn UI offers that must not survive a thread switch.

    ``fallback_prompt`` (the bounce-to-Auto offer) belongs to the turn that produced it; leaving it
    set across a New chat / open / delete would render the button in a DIFFERENT thread and re-run
    the wrong question. ``force_auto_prompt`` is popped every run, but clear it too for safety.
    """
    st.session_state.pop("fallback_prompt", None)
    st.session_state.pop("force_auto_prompt", None)


def _start_new_thread() -> None:
    st.session_state.thread_id = new_thread_id()
    st.session_state.thread_created_at = now_iso()
    st.session_state.messages = []
    st.session_state.agent_history = []
    _clear_transient_ui_state()


def _open_thread(thread_id: str) -> None:
    t = store.load(thread_id)
    st.session_state.thread_id = t.id
    st.session_state.thread_created_at = t.created_at
    st.session_state.messages = _deserialize_messages(t.display_messages)
    st.session_state.agent_history = t.agent_history
    _clear_transient_ui_state()


# ---- chat state (init before the sidebar, which lists threads) ----
if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()
    st.session_state.thread_created_at = now_iso()
if "messages" not in st.session_state:
    st.session_state.messages = []  # display history (user/assistant text + charts)
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []  # Anthropic-format message history

# ---- sidebar ----
@st.cache_data(ttl=300, show_spinner=False)
def _corpus_status() -> dict:
    """Cached corpus snapshot (active embedder + chunk count + freshness) for the sidebar badge."""
    return corpus_status(get_settings()).model_dump()


with st.sidebar:
    st.title("📈 Stock Research Agent")
    st.caption("Research / education only — **not financial advice**.")
    # Which embeddings filing questions are answered from (e.g. voyage-4) + corpus freshness.
    _cs = _corpus_status()
    _chunks = "unavailable" if _cs["chunks"] < 0 else f"{_cs['chunks']:,} chunks"
    st.caption(f"📚 **Filing search:** `{_cs['embedder']}` — {_chunks}")
    if _cs["latest"]:
        st.caption(f"{_cs['tickers']} tickers · fresh to {_cs['latest']}")
    st.divider()

    st.subheader("Quick starters")
    quick = {
        "📰 News summary": "Summarize the latest news for {ticker} and identify the key themes, risks, and catalysts.",
        "📊 Technical analysis": "Give me a technical analysis of {ticker}: trend, momentum, and volatility.",
        "🔮 Forecast 20 days": "What does the historical model forecast for {ticker} over the next 20 trading days?",
        "🔍 Full analysis": "Analyze {ticker} over the last 30 days — technicals, news summary, and 20-day forecast.",
    }
    ticker_input = st.text_input("Ticker", value="NVDA", max_chars=10).upper().strip()

    for label, template in quick.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending_prompt = template.replace("{ticker}", ticker_input)

    st.divider()
    # ---- Hybrid routing: Auto (LLM picks tools) vs. a deterministic domain ----
    st.subheader("Routing")
    route_mode = st.selectbox(
        "Mode",
        [_AUTO_MODE, *DOMAIN_NAMES],
        help=(
            "Auto lets the LLM choose and compose tools. A domain runs that one capability "
            "directly — no routing LLM call (faster/cheaper when you know what you want)."
        ),
    )
    # Selections used by the turn handler below (module scope — a `with` block adds no scope).
    selected_domain: str | None = None if route_mode == _AUTO_MODE else route_mode
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
        # Only surface the params the chosen domain actually uses (0 = use the tool's default).
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
        _save_current_thread()  # keep the one we're leaving
        _start_new_thread()
        st.rerun()

    for meta in store.list_threads():
        is_current = meta.id == st.session_state.thread_id
        row, trash = st.columns([0.82, 0.18])
        label = ("🟢 " if is_current else "") + meta.title
        if row.button(label, key=f"open_{meta.id}", use_container_width=True):
            _save_current_thread()
            _open_thread(meta.id)
            st.rerun()
        if trash.button("🗑️", key=f"del_{meta.id}", help="Delete this chat"):
            store.delete(meta.id)
            if is_current:
                _start_new_thread()
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

# ---- render history ----
st.title("Stock Research Agent")
st.caption(
    "Ask about any ticker — the agent calls tools for prices, indicators, news, forecasts, "
    "and SEC filings (single-shot or multi-hop)."
)

# Empty-state hero: discoverable capability showcase, shown only on a fresh chat.
# Auto-hides once a turn exists (the render loop below takes over).
if not st.session_state.messages:
    _render_capability_hero(ticker_input)

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Re-render any charts captured with this turn (survives Streamlit reruns).
        for spec in msg.get("charts", []):
            _render_chart(spec)
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))
            _render_export(msg["content"], msg.get("charts", []), idx)

# ---- handle quick-starter injection ----
pending = st.session_state.pop("pending_prompt", None)
# When a deterministic route can't handle the input (e.g. a full question typed into the theme
# box), the user can bounce the SAME text to the LLM router via an inline button. That sets
# `force_auto_prompt`, which overrides the sidebar routing mode for this one turn only.
force_auto_prompt = st.session_state.pop("force_auto_prompt", None)

# ---- input bar ----
if not settings.anthropic_api_key:
    st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

# Placeholder hints what to type for the active routing mode.
if selected_domain == "filings":
    _placeholder = "Type your filing question…"
elif selected_domain == "news" and selected_variant == "theme":
    _placeholder = "Type a sector/theme (e.g. robotics)…"
elif selected_domain is not None:
    _placeholder = f"Press enter to run {selected_domain} on {ticker_input or 'a ticker'}…"
else:
    _placeholder = "Ask anything about a stock…"

prompt = st.chat_input(_placeholder) or pending or force_auto_prompt
# A forced-Auto retry runs the LLM router regardless of the sidebar mode; otherwise the sidebar
# mode (Auto vs. a named domain) decides.
run_auto = selected_domain is None or (
    force_auto_prompt is not None and prompt == force_auto_prompt
)

if prompt:
    # Starting a fresh turn clears any pending bounce-to-Auto offer from a previous failed turn.
    st.session_state.pop("fallback_prompt", None)
    # Show user message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the request and render the response.
    with st.chat_message("assistant"):
        charts: list[ChartSpec] = []
        sources: list[dict] = []  # type: ignore[type-arg]
        spinner_msg = (
            "Working on your request…"
            if run_auto
            else f"Running {selected_domain}…"
        )
        with st.spinner(spinner_msg):
            if run_auto:
                # ---- Auto: the LLM agent loop picks + composes tools (unchanged). ----
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
                    # Charts derived from the tool results (numbers the tools produced,
                    # never the LLM) — rendered alongside the text below, not replacing it.
                    charts = charts_for(result.tool_results)
                    # Resolved SEC citations from the RAG tools (from tool output, not the LLM).
                    sources = _sources_from_tool_results(result.tool_results)
                    if result.tool_calls:
                        unique = list(dict.fromkeys(result.tool_calls))  # preserve order
                        st.caption(f"🔧 Tools used: {', '.join(unique)}")
                except AgentGroundingError as exc:
                    response = (
                        f"⚠️ The agent produced unverifiable figures and was stopped.\n\n"
                        f"Details: {exc}\n\n"
                        "Please rephrase your question or ask the agent to call a specific tool."
                    )
                except AgentError as exc:
                    low = str(exc).lower()
                    if "tim" in low or "connection" in low or "interrupt" in low:
                        response = (
                            "⚠️ The request timed out (often a temporary network hiccup or heavy "
                            "background load). Please try again — your conversation is preserved."
                        )
                    else:
                        response = f"⚠️ Agent error: {exc}"
            else:
                # ---- Deterministic: dispatch the chosen domain (no routing LLM call). ----
                # A deterministic turn is standalone (not part of the LLM dialogue), so
                # agent_history is left untouched. We wrap the single tool result in a
                # ToolInvocation to reuse the SAME chart + citation rendering as the Auto path.
                try:
                    route = resolve_domain(selected_domain, selected_variant)
                    rr = router.run(
                        prompt, route=route, ticker=ticker_input, horizon=horizon_val, days=days_val
                    )
                    structured = rr.structured or {}
                    tool_name = rr.tool_calls[0] if rr.tool_calls else route
                    invs = [ToolInvocation(name=tool_name, input={}, result=structured)]
                    charts = charts_for(invs)
                    sources = _sources_from_tool_results(invs)
                    if "error" in structured:
                        response = f"⚠️ {structured['error']}"
                        # LLM router may handle what this single route couldn't — offer the bounce.
                        st.session_state["fallback_prompt"] = prompt
                    elif "answer" in structured:
                        response = rr.text  # synthesis tools carry a cited prose answer
                    else:
                        # Numeric tools: show the structured result verbatim (charts render below).
                        response = (
                            f"**{selected_domain} · {selected_variant}** "
                            f"(`{tool_name}`)\n\n```json\n{rr.text}\n```"
                        )
                    st.caption(f"🎯 Deterministic route: {tool_name} (no LLM routing call)")
                except RouterError as exc:
                    response = f"⚠️ {exc}"
                    # e.g. a missing ticker/param the LLM router can infer from the text.
                    st.session_state["fallback_prompt"] = prompt
        st.markdown(response)
        for spec in charts:
            _render_chart(spec)
        _render_sources(sources)
        # Export buttons for this answer (idx = the index it gets once appended below).
        _render_export(response, charts, len(st.session_state.messages))

    # Persist text + charts + filing sources so the turn re-renders intact on later reruns.
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "charts": charts, "sources": sources}
    )
    # Save the thread to disk so it survives restarts and shows in the sidebar.
    _save_current_thread()

# ---- bounce-to-Auto affordance (rendered OUTSIDE the turn block so the click is captured) ----
# A Streamlit button must be re-instantiated on the run that handles its click; a button defined
# only inside `if prompt:` would be gone on the click rerun (prompt is empty then), so its click
# would be lost. We persist the failed text in `fallback_prompt` and render the button here every
# run until the user acts. Explicit (not automatic): a deterministic turn never silently spends a
# routing LLM call the user opted out of by choosing a domain.
_fallback = st.session_state.get("fallback_prompt")
if _fallback and st.button(
    "🤖 Ask this in Auto (LLM) mode instead",
    key="auto_fallback_btn",
    help="Re-run this exact question through the LLM router, which can compose multiple tools.",
):
    st.session_state["force_auto_prompt"] = _fallback
    st.session_state.pop("fallback_prompt", None)
    st.rerun()
