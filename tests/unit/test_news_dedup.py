"""Tests for near-duplicate detection and merging."""

from __future__ import annotations

from datetime import UTC, datetime

from stock_agent.news.dedup import deduplicate
from stock_agent.schemas.news import Article


def _article(
    title: str,
    url: str,
    *,
    summary: str | None = None,
    sentiment: float | None = None,
) -> Article:
    return Article(
        title=title,
        url=url,
        source="s",
        published_at=datetime(2025, 1, 2, tzinfo=UTC),
        summary=summary,
        sentiment=sentiment,
    )


def test_same_canonical_url_is_deduped() -> None:
    arts = [
        _article("Title A", "https://x.com/story?utm_source=feed"),
        _article("Different headline", "https://x.com/story/"),
    ]
    assert len(deduplicate(arts)) == 1


def test_near_duplicate_titles_merged() -> None:
    arts = [
        _article("Acme reports record quarterly revenue growth", "https://a.com/1"),
        _article("Acme reports record quarterly revenue growth today", "https://b.com/2"),
    ]
    assert len(deduplicate(arts)) == 1


def test_merge_fills_missing_summary_and_sentiment() -> None:
    arts = [
        _article("Acme posts strong earnings beat", "https://a.com/1"),
        _article(
            "Acme posts strong earnings beat",
            "https://a.com/1",
            summary="full detail",
            sentiment=0.4,
        ),
    ]
    out = deduplicate(arts)
    assert len(out) == 1
    assert out[0].summary == "full detail"
    assert out[0].sentiment == 0.4


def test_distinct_articles_kept() -> None:
    arts = [
        _article("Acme launches new chip", "https://a.com/1"),
        _article("Regulators open antitrust probe into Beta Corp", "https://b.com/2"),
    ]
    assert len(deduplicate(arts)) == 2
