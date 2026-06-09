"""Topic-news registry chain (failover) + TopicNewsFetcher (Enhancement C)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta

import pytest

from stock_agent.news.fetch import TopicNewsFetcher
from stock_agent.providers.base import ProviderError, ProviderUnavailable
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings


class FakeTopicProvider:
    """Minimal ``TopicNewsProvider`` double (structural match)."""

    def __init__(
        self,
        name: str,
        *,
        bundle: NewsBundle | None = None,
        raises: ProviderError | None = None,
        available: bool = True,
    ) -> None:
        self.name = name
        self._bundle = bundle
        self._raises = raises
        self._available = available

    def available(self) -> bool:
        return self._available

    def get_topic_news(
        self, keywords: Sequence[str], start: date, end: date, *, top_n: int = 25
    ) -> NewsBundle:
        if self._raises is not None:
            raise self._raises
        assert self._bundle is not None
        return self._bundle


def _bundle(n: int = 2) -> NewsBundle:
    arts = [
        Article(
            title=f"Robotics headline {i}",
            url=f"https://x.com/{i}",
            source="site.com",
            published_at=datetime.now(UTC) - timedelta(days=i),
        )
        for i in range(n)
    ]
    return NewsBundle(ticker="robotics", articles=arts)


def _registry(providers: list[FakeTopicProvider], priority: str) -> ProviderRegistry:
    return ProviderRegistry(
        providers, Settings(_env_file=None, provider_topic_priority=priority)
    )


def test_chain_returns_first_success() -> None:
    reg = _registry([FakeTopicProvider("a", bundle=_bundle())], "a")
    out = reg.get_topic_news(["robotics"], date(2026, 5, 1), date(2026, 6, 1))
    assert len(out) == 2


def test_chain_fails_over_to_next_provider() -> None:
    reg = _registry(
        [
            FakeTopicProvider("a", raises=ProviderUnavailable("a", "down")),
            FakeTopicProvider("b", bundle=_bundle(3)),
        ],
        "a,b",
    )
    out = reg.get_topic_news(["robotics"], date(2026, 5, 1), date(2026, 6, 1))
    assert len(out) == 3  # second provider served it


def test_no_topic_providers_raises() -> None:
    reg = _registry([], "gdelt_doc")
    with pytest.raises(ProviderUnavailable):
        reg.get_topic_news(["robotics"], date(2026, 5, 1), date(2026, 6, 1))


def test_fetcher_resolves_topic_and_ranks() -> None:
    reg = _registry([FakeTopicProvider("a", bundle=_bundle(2))], "a")
    resolved, bundle = TopicNewsFetcher(reg).fetch("robotics", lookback_days=14)
    assert resolved.known is True
    assert resolved.topic == "robotics"
    assert len(bundle) == 2
    assert bundle.ticker == "robotics"  # label carried as the bundle subject


def test_fetcher_freeform_topic() -> None:
    reg = _registry([FakeTopicProvider("a", bundle=_bundle(1))], "a")
    resolved, bundle = TopicNewsFetcher(reg).fetch("quantum computing")
    assert resolved.known is False
    assert resolved.keywords == ("quantum computing",)


class _KeywordLLM:
    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        return '{"keywords": ["iran war", "Iran-Israel conflict", "Iranian strikes"]}'


def test_fetcher_expands_freeform_topic_with_llm() -> None:
    # Free-form + expand + llm -> keywords widened (recall); curated themes untouched.
    reg = _registry([FakeTopicProvider("a", bundle=_bundle(1))], "a")
    resolved, _ = TopicNewsFetcher(reg).fetch("iran war", llm=_KeywordLLM(), expand=True)
    assert resolved.known is False
    assert "Iran-Israel conflict" in resolved.keywords
    assert resolved.keywords[0] == "iran war"


def test_fetcher_does_not_expand_known_theme() -> None:
    reg = _registry([FakeTopicProvider("a", bundle=_bundle(1))], "a")
    resolved, _ = TopicNewsFetcher(reg).fetch("robotics", llm=_KeywordLLM(), expand=True)
    assert resolved.keywords == ("robotics", "humanoid robot", "industrial automation")


class FakeToneProvider(FakeTopicProvider):
    """Topic provider that also exposes the optional get_topic_tone capability."""

    def __init__(self, name: str, tone: dict[str, object] | None) -> None:
        super().__init__(name, bundle=_bundle(1))
        self._tone = tone

    def get_topic_tone(
        self, keywords: Sequence[str], start: date, end: date
    ) -> dict[str, object] | None:
        return self._tone


def test_registry_topic_tone_from_supporting_provider() -> None:
    reg = _registry([FakeToneProvider("a", {"avg_tone": 2.0})], "a")
    assert reg.get_topic_tone(["x"], date(2026, 5, 1), date(2026, 6, 1)) == {"avg_tone": 2.0}


def test_registry_topic_tone_none_when_unsupported() -> None:
    # FakeTopicProvider has no get_topic_tone -> registry returns None (best-effort).
    reg = _registry([FakeTopicProvider("a", bundle=_bundle(1))], "a")
    assert reg.get_topic_tone(["x"], date(2026, 5, 1), date(2026, 6, 1)) is None
