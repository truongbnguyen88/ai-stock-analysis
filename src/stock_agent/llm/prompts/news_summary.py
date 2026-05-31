"""Versioned prompt for the Role A news summarizer.

The system text is static (so prompt caching applies); the user text carries the
per-ticker articles. Bump ``VERSION`` when the prompt changes materially.
"""

from __future__ import annotations

from stock_agent.schemas.news import Article

VERSION = "news_summary.v3"

# Static instructions — cached. Encodes the numbers-vs-narrative invariant and
# the exact JSON contract (mirrors stock_agent.llm.guards.NewsSummary).
SYSTEM = """You are a financial news analyst assistant. Your ONLY job is qualitative \
synthesis of the provided news articles for a single stock ticker.

STRICT RULES:
- Summarize and interpret the articles' content. Do NOT give investment advice and \
do NOT make buy/sell/hold recommendations.
- You MUST NOT invent your own probabilities, odds, likelihoods, or numeric \
forecasts of future returns. Probabilities come only from statistical models \
elsewhere in the system, never from you.
- You MAY report concrete facts and figures that appear in the articles \
(e.g. "revenue rose 20%", "analyst raised price target to $300", "EPS of $1.87"). \
Reporting what articles say is allowed; inventing your own forward-looking numbers \
is not.
- Do NOT set your own price targets. Do NOT say "the stock will reach $X" or \
"I expect a 30% upside." You may report that analysts set or raised price targets.
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


# Critique rubric for the reflection pass. The model re-reads the articles and
# its own draft, then revises. The goal is depth without padding — never invent
# points, citations, or forecasts to make the summary look fuller.
_REFLECTION_RUBRIC = """\
This is a SELF-REVIEW pass. Below are the same articles and your own DRAFT summary. \
Critique the draft and return an IMPROVED JSON object in the identical shape. Check:
- Completeness: did you miss a material theme, driver, risk, or catalyst the articles support?
- Balance: are bullish AND bearish AND risks represented WHEN the articles support them \
(do not force a side that the news does not back)?
- Evidence: does every point cite at least one of the provided URLs? Drop or merge any \
point that no article supports.
- Specificity: prefer concrete facts (named products, figures stated IN the articles, \
dates) over vague phrasing; merge redundant points.
- Hard rules (unchanged): no probabilities, odds, likelihoods, your own price targets, or \
numeric forecasts of future returns; cite ONLY the provided URLs.
If the draft is already strong, return it lightly refined — do NOT pad, invent, or \
speculate to make it longer. Return ONLY the JSON object."""


def build_reflection(ticker: str, articles: list[Article], draft_json: str) -> str:
    """Render the reflection user message: articles + the draft + the critique rubric.

    Reuses the same SYSTEM contract (so prompt caching still hits); only the user
    text differs from the first pass.
    """
    return (
        build_user(ticker, articles)
        + "\n\n=== YOUR DRAFT SUMMARY (to review and improve) ===\n"
        + draft_json
        + "\n\n"
        + _REFLECTION_RUBRIC
    )
