"""Role A news summarizer: ranked articles -> guarded ``NewsSummary``.

Flow: build prompt -> LLM JSON -> parse -> anti-forecast guard (one corrective
retry) -> strip fabricated citations. The summarizer never produces numbers; if
the model insists on forecast language after a retry, it raises rather than
emitting a violation.
"""

from __future__ import annotations

import json
from typing import Any

from stock_agent.llm.client import TextLLM
from stock_agent.llm.guards import (
    NewsSummary,
    find_forecast_violations,
    sanitize_citations,
)
from stock_agent.llm.prompts.news_summary import SYSTEM, build_user
from stock_agent.logging_config import get_logger
from stock_agent.news.clean import canonical_url
from stock_agent.schemas.news import NewsBundle

log = get_logger(__name__)


class SummaryGuardError(RuntimeError):
    """Raised when the LLM keeps producing forecast language after a retry."""


def _loads_lenient(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating leading/trailing prose around it."""
    try:
        result: Any = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        result = json.loads(text[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("LLM output was not a JSON object")
    return result


def _parse(raw: str) -> NewsSummary:
    return NewsSummary.model_validate(_loads_lenient(raw))


def summarize_news(
    bundle: NewsBundle, llm: TextLLM, *, max_articles: int = 15, max_tokens: int = 4096
) -> NewsSummary:
    """Summarize the top articles of ``bundle`` into a guarded ``NewsSummary``."""
    articles = bundle.articles[:max_articles]
    allowed = {canonical_url(str(a.url)) for a in articles}
    user = build_user(bundle.ticker, articles)

    summary = _parse(llm.complete_json(system=SYSTEM, user=user, max_tokens=max_tokens))

    violations = find_forecast_violations(summary)
    if violations:
        log.warning("llm.forecast_violation", ticker=bundle.ticker, n=len(violations))
        correction = (
            user + "\n\nIMPORTANT: your previous answer contained forbidden forecast/probability "
            "language (e.g.: "
            + "; ".join(violations[:3])
            + "). Rewrite WITHOUT any probabilities, odds, likelihoods, price targets, or "
            "numeric forecasts of future returns."
        )
        summary = _parse(llm.complete_json(system=SYSTEM, user=correction, max_tokens=max_tokens))
        remaining = find_forecast_violations(summary)
        if remaining:
            raise SummaryGuardError(
                f"forecast language persisted after retry: {'; '.join(remaining[:3])}"
            )

    # Drop any fabricated citations (keep the point text).
    return sanitize_citations(summary, allowed)
