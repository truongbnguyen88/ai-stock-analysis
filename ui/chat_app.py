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

from stock_agent.agent.runtime import AgentError, AgentGroundingError, AnthropicToolClient, run_agent
from stock_agent.agent.tools import ToolExecutor
from stock_agent.llm.client import AnthropicClient
from stock_agent.settings import get_settings

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

    st.session_state.messages.append({"role": "assistant", "content": response})
