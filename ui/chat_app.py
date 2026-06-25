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

from stock_agent.agent.runtime import AgentError, AgentGroundingError, AnthropicToolClient, run_agent
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
from stock_agent.settings import get_settings
from stock_agent.reports.export import EXPORT_META, export_summary
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


def _start_new_thread() -> None:
    st.session_state.thread_id = new_thread_id()
    st.session_state.thread_created_at = now_iso()
    st.session_state.messages = []
    st.session_state.agent_history = []


def _open_thread(thread_id: str) -> None:
    t = store.load(thread_id)
    st.session_state.thread_id = t.id
    st.session_state.thread_created_at = t.created_at
    st.session_state.messages = _deserialize_messages(t.display_messages)
    st.session_state.agent_history = t.agent_history


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
if "pending_prompt" in st.session_state:
    pending = st.session_state.pop("pending_prompt")
else:
    pending = None

# ---- input bar ----
if not settings.anthropic_api_key:
    st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

prompt = st.chat_input("Ask anything about a stock…") or pending

if prompt:
    # Show user message immediately.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run the agent and stream the response.
    with st.chat_message("assistant"):
        charts: list[ChartSpec] = []
        sources: list[dict] = []  # type: ignore[type-arg]
        with st.spinner("Working on your request…"):
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
                # Show which tools were used as a subtle annotation.
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
