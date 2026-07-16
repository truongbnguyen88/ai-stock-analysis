"""Pipeline seam: NewsSummary + NewsBundle -> NewsAnalysis (`_news_analysis`).

Locks the one integration point the memo/tool tests mock out: the flattening of the LLM
summary plus the free provider-sentiment shares (whose dict keys come from
``features.news_features``) into the pure ``NewsAnalysis`` schema.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from stock_agent.llm.guards import CitedPoint, NewsSummary
from stock_agent.pipelines.research import _news_analysis
from stock_agent.schemas.news import Article, NewsBundle


def _article(sentiment: float | None) -> Article:
    return Article(
        title="NVDA news",
        url="https://example.com/a",
        source="wire",
        published_at=datetime(2026, 7, 1, 12, 0, 0),
        sentiment=sentiment,
    )


def _summary() -> NewsSummary:
    return NewsSummary(
        overview="AI demand dominates.",
        key_themes=["AI capex", "supply"],
        bullish=[CitedPoint(point="Capex guidance raised", sources=["u1"])],
        bearish=[CitedPoint(point="Export curbs widened", sources=["u2"])],
        risks=[CitedPoint(point="Concentration", sources=["u3"])],
        catalysts=[CitedPoint(point="GTC keynote", sources=["u4"])],
    )


def test_news_analysis_flattens_summary_and_scores() -> None:
    # 3 scored (+0.5, -0.5, +0.2) + 1 unscored → coverage 3/4; pos {+0.5,+0.2}, neg {-0.5}.
    bundle = NewsBundle(
        ticker="NVDA",
        articles=[_article(0.5), _article(-0.5), _article(0.2), _article(None)],
    )
    na = _news_analysis(_summary(), bundle, lookback_days=21)

    assert na.lookback_days == 21
    assert na.article_count == 4
    assert na.overview == "AI demand dominates."
    assert na.key_themes == ["AI capex", "supply"]
    # CitedPoint.point is flattened to plain strings.
    assert na.bullish == ["Capex guidance raised"]
    assert na.risks == ["Concentration"] and na.catalysts == ["GTC keynote"]
    # Shares are fractions of ALL articles (unscored count toward neither).
    assert na.sentiment_coverage == 0.75
    assert na.pct_positive == 0.5  # 2 of 4 above +0.15
    assert na.pct_negative == 0.25  # 1 of 4 below -0.15
    # Mean over the 3 SCORED articles: (0.5 - 0.5 + 0.2) / 3.
    assert na.avg_sentiment == pytest.approx(0.2 / 3)


def test_news_analysis_none_shares_when_no_scores() -> None:
    bundle = NewsBundle(ticker="NVDA", articles=[_article(None), _article(None)])
    na = _news_analysis(_summary(), bundle, lookback_days=21)
    # No provider scores → shares are None (chart no-ops), not a fake all-neutral 0.0.
    assert na.pct_positive is None and na.pct_negative is None
    assert na.avg_sentiment is None  # no fabricated 0.0 average either
    assert na.sentiment_coverage == 0.0
    assert na.article_count == 2


def test_news_analysis_handles_empty_bundle() -> None:
    na = _news_analysis(_summary(), NewsBundle(ticker="NVDA", articles=[]), lookback_days=21)
    assert na.article_count == 0
    assert na.pct_positive is None and na.pct_negative is None
    # Qualitative insights still flow through from the summary.
    assert na.bullish == ["Capex guidance raised"]
