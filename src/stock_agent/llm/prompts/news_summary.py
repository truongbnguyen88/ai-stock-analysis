"""Versioned prompt for the Role A news summarizer.

The system text is static (so prompt caching applies); the user text carries the
per-ticker articles. Bump ``VERSION`` when the prompt changes materially.
"""

from __future__ import annotations

from stock_agent.schemas.news import Article

VERSION = "news_summary.v1"

# Static instructions — cached. Encodes the numbers-vs-narrative invariant and
# the exact JSON contract (mirrors stock_agent.llm.guards.NewsSummary).
SYSTEM = """You are a financial news analyst assistant. Your ONLY job is qualitative \
synthesis of the provided news articles for a single stock ticker.

STRICT RULES:
- Summarize and interpret the articles' content. Do NOT give investment advice and \
do NOT make buy/sell/hold recommendations.
- You MUST NOT invent or state any probabilities, odds, likelihoods, price targets, \
or numeric forecasts of future returns. Probabilities come only from statistical \
models elsewhere in the system, never from you. You may report concrete facts that \
appear in the articles (e.g. "revenue rose 20%"), but never forward-looking numbers.
- Use ONLY the article URLs provided below as sources. Never invent URLs.
- Base everything on the supplied articles; if evidence is thin, say so briefly in \
the overview rather than speculating.

OUTPUT: Return ONLY a single JSON object (no prose, no markdown) with this shape:
{
  "overview": "<2-4 sentence neutral synthesis>",
  "key_themes": ["<short theme>", ...],
  "bullish": [{"point": "<positive development>", "sources": ["<url>", ...]}, ...],
  "bearish": [{"point": "<negative development>", "sources": ["<url>", ...]}, ...],
  "risks": [{"point": "<risk factor>", "sources": ["<url>", ...]}, ...],
  "catalysts": [{"point": "<upcoming catalyst/event>", "sources": ["<url>", ...]}, ...]
}
Each "sources" entry must be one of the provided article URLs. Use [] when no \
specific article supports a point. Keep points concise and grounded."""


def build_user(ticker: str, articles: list[Article]) -> str:
    """Render the per-request user message: the ticker + the ranked articles."""
    lines = [f"Ticker: {ticker}", "", "Articles (most relevant first):", ""]
    for i, article in enumerate(articles, start=1):
        published = article.published_at.date().isoformat()
        lines.append(f"[{i}] {article.title} — {article.source} ({published})")
        lines.append(f"    url: {article.url}")
        if article.summary:
            lines.append(f"    summary: {article.summary}")
        lines.append("")
    lines.append("Synthesize these into the required JSON object, citing only the URLs above.")
    return "\n".join(lines)
