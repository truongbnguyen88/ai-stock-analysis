"""Near-duplicate detection for merged news.

Merging Finnhub + Marketaux + Alpha Vantage produces overlap: the same story
appears multiple times (often syndicated under different URLs). Two articles are
duplicates if they share a canonical URL OR their titles are highly similar
(token Jaccard >= threshold). The kept copy absorbs missing fields from its
duplicates (summary, sentiment, topics).
"""

from __future__ import annotations

import re

from stock_agent.news.clean import canonical_url
from stock_agent.schemas.news import Article

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_DEFAULT_THRESHOLD = 0.85


def _title_tokens(title: str) -> set[str]:
    """Lowercase, strip punctuation, return the set of word tokens."""
    normalized = _WS.sub(" ", _NON_ALNUM.sub(" ", title.lower())).strip()
    return set(normalized.split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two token sets (0..1)."""
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _is_duplicate(a: Article, b: Article, *, threshold: float) -> bool:
    if canonical_url(str(a.url)) == canonical_url(str(b.url)):
        return True
    return _jaccard(_title_tokens(a.title), _title_tokens(b.title)) >= threshold


def _merge(keep: Article, dup: Article) -> Article:
    """Fill fields missing on ``keep`` from ``dup`` (prefer richer metadata)."""
    update: dict[str, object] = {}
    if not keep.summary and dup.summary:
        update["summary"] = dup.summary
    if keep.sentiment is None and dup.sentiment is not None:
        update["sentiment"] = dup.sentiment
    if not keep.topics and dup.topics:
        update["topics"] = dup.topics
    return keep.model_copy(update=update) if update else keep


def deduplicate(
    articles: list[Article], *, title_threshold: float = _DEFAULT_THRESHOLD
) -> list[Article]:
    """Return articles with near-duplicates merged away (stable order preserved)."""
    kept: list[Article] = []
    for candidate in articles:
        for i, existing in enumerate(kept):
            if _is_duplicate(existing, candidate, threshold=title_threshold):
                kept[i] = _merge(existing, candidate)
                break
        else:
            kept.append(candidate)
    return kept
