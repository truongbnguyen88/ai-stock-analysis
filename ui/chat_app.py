"""Streamlit chat frontend for the stock research agent.

Launch with:
    streamlit run ui/chat_app.py

Requires ANTHROPIC_API_KEY (and optionally news provider keys) in .env.
This is NOT financial advice.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the repo root without installing in editable mode.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st

# ---- page config (must be the first Streamlit call) ----
st.set_page_config(
    page_title="Stock Research Agent",
    page_icon="📈",
    layout="wide",
)

import altair as alt
import pandas as pd

from stock_agent.agent.runtime import AgentError, AgentGroundingError, AnthropicToolClient, run_agent
from stock_agent.agent.tools import ToolExecutor
from stock_agent.llm.client import AnthropicClient
from stock_agent.settings import get_settings
from stock_agent.viz.charts import ChartSpec, charts_for


def _render_chart(spec: ChartSpec) -> None:
    """Render one ChartSpec as an Altair chart alongside the agent's text."""
    st.markdown(f"**{spec.title}**")
    y_axis = alt.Axis(format="%") if spec.y_is_percent else alt.Axis()
    y_enc = alt.Y(f"{spec.y}:Q", axis=y_axis, title=None)
    x_sort = list(spec.x_sort) if spec.x_sort else "ascending"

    if spec.kind == "reliability":
        # Predicted vs realized, with the y=x ideal as a dashed reference.
        base = alt.Chart(spec.data)
        pts = base.mark_circle(size=90, color="#4c78a8").encode(
            x=alt.X("predicted:Q", scale=alt.Scale(domain=[0, 1]), title="Predicted"),
            y=alt.Y("realized:Q", scale=alt.Scale(domain=[0, 1]), title="Realized"),
        )
        ideal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(strokeDash=[5, 5], color="gray")
            .encode(x="x:Q", y="y:Q")
        )
        chart: alt.Chart = (ideal + pts)
    elif spec.kind == "grouped_bar":
        chart = alt.Chart(spec.data).mark_bar().encode(
            x=alt.X(f"{spec.x}:N", sort=x_sort, title=None),
            y=y_enc,
            color=alt.Color(f"{spec.color}:N", title=None),
            xOffset=f"{spec.color}:N",
            tooltip=list(spec.data.columns),
        )
    else:  # "bar"
        chart = alt.Chart(spec.data).mark_bar(color="#4c78a8").encode(
            x=alt.X(f"{spec.x}:N", sort=x_sort, title=None),
            y=y_enc,
            tooltip=list(spec.data.columns),
        )

    st.altair_chart(chart.properties(height=260), use_container_width=True)
    if spec.caption:
        st.caption(spec.caption)

# ---- load settings once ----
@st.cache_resource
def load_resources():
    """Build shared (cached) agent resources — one set per Streamlit session."""
    settings = get_settings()
    executor = ToolExecutor(settings, llm=AnthropicClient(settings))
    llm = AnthropicToolClient(settings)
    return settings, executor, llm

settings, executor, agent_llm = load_resources()

# ---- sidebar ----
with st.sidebar:
    st.title("📈 Stock Research Agent")
    st.caption("Research / education only — **not financial advice**.")
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
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_history = []  # also reset conversation memory

    st.divider()
    st.caption(
        "**Keys configured:**\n"
        + ("✅ Anthropic\n" if settings.anthropic_api_key else "❌ Anthropic (required)\n")
        + ("✅ Finnhub\n" if settings.finnhub_api_key else "⬜ Finnhub\n")
        + ("✅ Marketaux\n" if settings.marketaux_api_key else "⬜ Marketaux\n")
        + ("✅ Alpha Vantage" if settings.alpha_vantage_api_key else "⬜ Alpha Vantage")
    )

# ---- chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = []       # display history (user/assistant text)
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []  # Anthropic-format message history

# ---- render history ----
st.title("Stock Research Agent")
st.caption("Ask about any ticker — the agent calls tools for prices, indicators, news, and forecasts.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Re-render any charts captured with this turn (survives Streamlit reruns).
        for spec in msg.get("charts", []):
            _render_chart(spec)

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
        with st.spinner("Calling tools…"):
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
                response = f"⚠️ Agent error: {exc}"
        st.markdown(response)
        for spec in charts:
            _render_chart(spec)

    # Persist text + charts so the turn re-renders intact on later reruns.
    st.session_state.messages.append({"role": "assistant", "content": response, "charts": charts})
