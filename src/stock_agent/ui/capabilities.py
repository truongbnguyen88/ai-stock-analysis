"""Capability showcase content — single source of truth for the launch screen.

Data-driven (NOT a tool dump): each ``Capability`` is a value statement the agent
can deliver, plus a ready-to-run ``example`` prompt. The Streamlit empty-state
hero renders these as both the typewriter headlines and the interactive cards
(see ``ui/chat_app.py``), so the visual and the action read from one list.

Pure data, no I/O — unit-tested for well-formedness (every ``example`` carries a
``{ticker}`` placeholder the UI fills from the sidebar ticker).
"""

from __future__ import annotations

from dataclasses import dataclass

TICKER_PLACEHOLDER = "{ticker}"


@dataclass(frozen=True)
class Capability:
    """One advertised capability: display copy + a runnable example prompt."""

    icon: str  # emoji shown on the card / typewriter line
    title: str  # short value statement, e.g. "Probabilistic forecasts"
    blurb: str  # one-line "what you get", e.g. "Calibrated up/down odds + VaR/CI"
    example: str  # runnable prompt; MUST contain TICKER_PLACEHOLDER ("{ticker}")

    def render_example(self, ticker: str) -> str:
        """Fill the example prompt with a concrete ticker for submission."""
        return self.example.replace(TICKER_PLACEHOLDER, ticker)


# Curated value statements (not tool names). Ordered roughly by how a user would
# explore: understand the name → read its filings → forecast it → sanity-check the
# forecast → read/synthesize the news → get the integrated brief → widen to
# multi-ticker/themes. Each example routes to a real agent capability.
CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        icon="📊",
        title="Technical analysis",
        blurb="Trend, momentum, volatility — MAs, RSI, MACD, ATR, drawdown",
        example="Give me a technical analysis of {ticker}: trend, momentum, and volatility.",
    ),
    Capability(
        icon="📄",
        title="Ask the SEC filings",
        blurb="Risk factors, MD&A, business drivers — answered from the 10-K/10-Q, cited",
        example=(
            "What are {ticker}'s key risk factors, and what does management say about "
            "demand and margins? Use the SEC filings and cite them."
        ),
    ),
    Capability(
        icon="🔗",
        title="Multi-hop filing research",
        blurb="Compare filings, track changes, follow the trail across documents — cited",
        example=(
            "Compare {ticker}'s and AMD's risk factors across their SEC filings — where "
            "do they overlap and differ? Cite the filings."
        ),
    ),
    Capability(
        icon="🔮",
        title="Probabilistic forecasts",
        blurb="Calibrated up/down odds, expected move, VaR & CI — 5–60 days",
        example="Forecast {ticker} 30 days out with the ensemble model.",
    ),
    Capability(
        icon="🎯",
        title="Is this forecast trustworthy?",
        blurb="Out-of-sample calibration (ECE) and walk-forward track record",
        example="Is the 20-day forecast for {ticker} well-calibrated? Show the backtest.",
    ),
    Capability(
        icon="⚡",
        title="Chance of a big move",
        blurb="P(|return| > k) split into up- and down-tails — the ML model's edge",
        example="What's the chance of a big move in {ticker} over the next 30 days?",
    ),
    Capability(
        icon="📰",
        title="Latest news, newest-first",
        blurb="Recent headlines with real sources and links",
        example="Show me the latest news for {ticker}.",
    ),
    Capability(
        icon="🧭",
        title="News synthesis",
        blurb="Themes, risks, and catalysts from the news — cited, qualitative",
        example="Summarize {ticker} news and pull out the key themes, risks, and catalysts.",
    ),
    Capability(
        icon="🧩",
        title="Executive research brief",
        blurb="Filings + news + forecast fused into one cited, non-advisory summary",
        example="Give me the full executive research brief on {ticker}.",
    ),
    Capability(
        icon="⚖️",
        title="Compare multiple tickers",
        blurb="Side-by-side news sentiment and model forecasts across names",
        example="Compare the 20-day forecasts and news sentiment for {ticker}, MSFT, and AMD.",
    ),
    Capability(
        icon="🛰️",
        title="Analyze a theme's news",
        blurb="Pull news by topic — robotics, EVs, AI memory, semiconductors…",
        example="Pull and analyze recent news about robotics.",
    ),
)
