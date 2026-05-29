"""Tests for news cleaning + URL canonicalization."""

from __future__ import annotations

from datetime import datetime

from stock_agent.news.clean import canonical_url, clean_article, clean_articles
from stock_agent.schemas.news import Article


def _article(title: str = "T", url: str = "https://x.com/a", summary: str | None = None) -> Article:
    return Article(
        title=title, url=url, source="src", published_at=datetime(2025, 1, 2), summary=summary
    )


def test_canonical_url_strips_tracking_fragment_and_trailing_slash() -> None:
    raw = "HTTPS://Example.com/Path/?utm_source=x&id=7#frag"
    assert canonical_url(raw) == "https://example.com/Path?id=7"


def test_canonical_url_equates_tracking_variants() -> None:
    a = canonical_url("https://x.com/story?utm_campaign=a")
    b = canonical_url("https://x.com/story/")
    assert a == b == "https://x.com/story"


def test_clean_article_trims_and_makes_tz_aware() -> None:
    art = _article(title="  Big   News  ", summary="  hi  ")
    cleaned = clean_article(art)
    assert cleaned.title == "Big News"
    assert cleaned.summary == "hi"
    assert cleaned.published_at.tzinfo is not None  # coerced to UTC


def test_clean_articles_drops_empty_titles() -> None:
    arts = [_article(title="   "), _article(title="Real")]
    out = clean_articles(arts)
    assert len(out) == 1 and out[0].title == "Real"
