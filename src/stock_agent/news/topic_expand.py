"""LLM keyword expansion for free-form topic queries (Enhancement C — precision).

A free-form theme (e.g. "iran war") is searched as a single exact phrase, which
misses near-synonyms and named entities ("Iran-Israel conflict", "Iranian
strikes"). This expands the phrase into a small set of OR-able search keywords.

These are SEARCH terms (narrative-side), never numbers — so this does not touch
the numbers-vs-narrative invariant. Best-effort: any failure returns just the
original phrase (safe no-op), so search precision never depends on the LLM.
"""

from __future__ import annotations

import json

from stock_agent.llm.client import TextLLM
from stock_agent.logging_config import get_logger

log = get_logger(__name__)

_SYSTEM = """You expand a news-search topic into precise search keywords.
Return ONLY a JSON object: {"keywords": ["...", "..."]}.
Rules: 3-6 short keyword phrases a news search engine would match for the topic;
include close synonyms and the key named entities; English; no explanations."""


def expand_topic_keywords(phrase: str, llm: TextLLM, *, max_keywords: int = 6) -> tuple[str, ...]:
    """Expand a free-form topic phrase into OR-able search keywords via the LLM.

    Returns the original phrase first, then model-suggested synonyms/entities
    (deduped case-insensitively, capped at ``max_keywords``). On any failure or
    empty response, returns just ``(phrase,)`` — search still works, just narrower.
    """
    phrase = phrase.strip()
    try:
        raw = llm.complete_json(system=_SYSTEM, user=f"Topic: {phrase}", max_tokens=256)
        parsed = json.loads(raw)
        items = parsed.get("keywords", []) if isinstance(parsed, dict) else []
    except (json.JSONDecodeError, AttributeError, ValueError, TypeError):
        items = []

    out: list[str] = [phrase]
    seen: set[str] = {phrase.lower()}
    for item in items:
        term = str(item).strip()
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
        if len(out) >= max_keywords:
            break
    log.info("topic.keyword_expansion", phrase=phrase, n_keywords=len(out))
    return tuple(out)
