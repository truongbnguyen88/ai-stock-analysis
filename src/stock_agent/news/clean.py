"""Article metadata normalization.

Cleans text fields, canonicalizes URLs (so duplicates from different providers
collapse), and coerces naive timestamps to UTC. Pure functions; no I/O.
"""

from __future__ import annotations

import re
from datetime import UTC
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from stock_agent.schemas.news import Article

# Query-param prefixes that are tracking noise, not content identity.
_TRACKING_PREFIXES = ("utm_", "fbclid", "gclid", "mc_", "ref", "spm")
_WS = re.compile(r"\s+")


def canonical_url(url: str) -> str:
    """Return a normalized URL for dedup keying.

    Lowercases scheme/host, drops the fragment, strips tracking query params,
    and removes a trailing slash. Preserves meaningful path/query.
    """
    parts = urlsplit(url)
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not any(k.lower().startswith(p) for p in _TRACKING_PREFIXES)
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def _clean_text(text: str) -> str:
    """Collapse internal whitespace and trim."""
    return _WS.sub(" ", text).strip()


def clean_article(article: Article) -> Article:
    """Return a cleaned copy: trimmed text + tz-aware (UTC) timestamp."""
    update: dict[str, object] = {
        "title": _clean_text(article.title),
        "source": _clean_text(article.source),
        "summary": _clean_text(article.summary) if article.summary else None,
    }
    # Treat naive timestamps as UTC so downstream comparisons are unambiguous.
    if article.published_at.tzinfo is None:
        update["published_at"] = article.published_at.replace(tzinfo=UTC)
    return article.model_copy(update=update)


def clean_articles(articles: list[Article]) -> list[Article]:
    """Clean all articles, dropping any with an empty title after cleaning."""
    out: list[Article] = []
    for article in articles:
        if not _clean_text(article.title):
            continue
        out.append(clean_article(article))
    return out
