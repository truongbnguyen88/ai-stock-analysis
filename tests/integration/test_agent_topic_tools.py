"""Topic/theme news agent tools (Enhancement C) — get_topic_news / analyze_topic_news.

Offline + deterministic: a fake TopicNewsProvider serves articles and a FakeLLM
serves a canned summary, so the resolve -> fetch -> synthesize path runs without
any network.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

from stock_agent.agent.tools import ToolExecutor
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings


class FakeTopicProvider:
    name = "gdelt_doc"

    def __init__(self, bundle: NewsBundle) -> None:
        self._bundle = bundle

    def available(self) -> bool:
        return True

    def get_topic_news(
        self, query: str, start: date, end: date, *, top_n: int = 25
    ) -> NewsBundle:
        return self._bundle


class FakeLLM:
    """Returns a canned summary JSON for every complete_json call."""

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        self.calls += 1
        return self._payload


_ARTICLE_URL = "https://site.com/robotics"


def _bundle(n: int = 2) -> NewsBundle:
    arts = [
        Article(
            title=f"Humanoid robot milestone {i}",
            url=f"{_ARTICLE_URL}/{i}",
            source="site.com",
            published_at=datetime.now(UTC) - timedelta(days=i),
        )
        for i in range(n)
    ]
    return NewsBundle(ticker="robotics", articles=arts)


def _summary_payload() -> str:
    return json.dumps(
        {
            "overview": "Robotics startups are scaling humanoid platforms.",
            "key_themes": ["humanoid robots", "automation"],
            "bullish": [{"point": "New funding rounds", "sources": [f"{_ARTICLE_URL}/0"]}],
            "bearish": [],
            "risks": [],
            "catalysts": [],
        }
    )


def _executor(*, llm: FakeLLM | None = None, bundle: NewsBundle | None = None) -> ToolExecutor:
    provider = FakeTopicProvider(bundle if bundle is not None else _bundle())
    registry = ProviderRegistry(
        [provider], Settings(_env_file=None, provider_topic_priority="gdelt_doc")
    )
    return ToolExecutor(Settings(_env_file=None), registry=registry, llm=llm)


# ---- get_topic_news ----------------------------------------------------------
def test_get_topic_news_returns_resolved_keywords_and_headlines() -> None:
    r = _executor().execute("get_topic_news", {"topic": "robotics", "days": 14})
    assert "error" not in r
    assert r["known_topic"] is True
    assert "robotics" in r["keywords"]
    assert r["resolved_query"].startswith("(")  # multi-keyword OR expression
    assert len(r["articles"]) == 2
    assert r["articles"][0]["url"].startswith(_ARTICLE_URL)


def test_get_topic_news_freeform_theme() -> None:
    r = _executor().execute("get_topic_news", {"topic": "quantum computing"})
    assert r["known_topic"] is False
    assert r["keywords"] == ["quantum computing"]


# ---- analyze_topic_news ------------------------------------------------------
def test_analyze_topic_news_synthesizes_and_reports_transparency() -> None:
    llm = FakeLLM(_summary_payload())
    r = _executor(llm=llm).execute("analyze_topic_news", {"topic": "robotics", "days": 14})
    assert "error" not in r
    assert r["overview"].startswith("Robotics startups")
    assert r["resolved_query"].startswith("(")
    assert r["article_count"] == 2
    # No per-article tone from this source -> sentiment honestly reported as None.
    assert r["topic_sentiment"] is None
    assert r["bullish"][0]["sources"] == [f"{_ARTICLE_URL}/0"]  # citation preserved


def test_analyze_topic_news_without_llm_errors() -> None:
    r = _executor(llm=None).execute("analyze_topic_news", {"topic": "robotics"})
    assert "error" in r and "no LLM" in r["error"]


def test_analyze_topic_news_empty_bundle_errors_gracefully() -> None:
    llm = FakeLLM(_summary_payload())
    r = _executor(llm=llm, bundle=NewsBundle(ticker="robotics", articles=[])).execute(
        "analyze_topic_news", {"topic": "robotics"}
    )
    assert r["error"] == "no news found for this theme in the window"
    assert r["known_topic"] is True  # transparency block still present
