"""Registry wiring for the new news providers — no live calls (availability + chain resolution)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from stock_agent.providers.base import NewsProvider, ProviderRateLimit, TopicNewsProvider
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings

_NEW = {"fmp", "guardian", "newsdata", "google_news_rss"}


class _FakeTopic:
    """In-process TopicNewsProvider: returns a bundle, or raises, on get_topic_news."""

    def __init__(self, name: str, bundle: NewsBundle | None = None, error: Exception | None = None):
        self.name = name
        self._bundle = bundle
        self._error = error

    def available(self) -> bool:
        return True

    def get_topic_news(self, keywords, start, end, *, top_n=25):  # type: ignore[no-untyped-def]
        if self._error is not None:
            raise self._error
        assert self._bundle is not None
        return self._bundle


def _bundle(n: int) -> NewsBundle:
    arts = [
        Article(title=f"a{i}", url=f"https://x.com/{i}", source="s",
                published_at=datetime(2026, 6, 1, tzinfo=UTC))
        for i in range(n)
    ]
    return NewsBundle(ticker="t", articles=arts)


def _topic_registry(*providers: _FakeTopic) -> ProviderRegistry:
    names = ",".join(p.name for p in providers)
    settings = Settings(_env_file=None, provider_topic_priority=names)
    return ProviderRegistry(list(providers), settings)


def test_all_new_providers_registered() -> None:
    reg = build_default_registry(Settings(_env_file=None))
    assert set(reg._by_name) >= _NEW


def test_priority_strings_reference_only_registered_providers() -> None:
    # Guards against a typo in a comma-separated priority default silently dropping a provider.
    settings = Settings(_env_file=None)
    reg = build_default_registry(settings)
    known = set(reg._by_name)
    assert set(settings.news_priority) <= known, set(settings.news_priority) - known
    assert set(settings.topic_priority) <= known, set(settings.topic_priority) - known


def test_no_keys_resolves_only_keyless_sources() -> None:
    # With no API keys, only the keyless providers (gdelt_doc, google_news_rss) survive chain
    # resolution; the keyed ones are skipped via available()==False.
    settings = Settings(_env_file=None)
    reg = build_default_registry(settings)
    topic = [p.name for p in reg._chain(settings.topic_priority, TopicNewsProvider)]  # type: ignore[type-abstract]
    news = [p.name for p in reg._chain(settings.news_priority, NewsProvider)]  # type: ignore[type-abstract]
    assert topic == ["gdelt_doc", "google_news_rss"]  # guardian/marketaux/newsdata need keys
    assert news == ["google_news_rss"]  # finnhub/fmp/marketaux/av/newsdata need keys
    # Keyless company news + topic news therefore work out of the box.


def test_keys_activate_the_keyed_providers() -> None:
    settings = Settings(
        _env_file=None,
        finnhub_api_key="a", marketaux_api_key="b", alpha_vantage_api_key="c",
        fmp_api_key="d", guardian_api_key="e", newsdata_api_key="f",
    )
    reg = build_default_registry(settings)
    news = [p.name for p in reg._chain(settings.news_priority, NewsProvider)]  # type: ignore[type-abstract]
    topic = [p.name for p in reg._chain(settings.topic_priority, TopicNewsProvider)]  # type: ignore[type-abstract]
    # fmp is intentionally absent from the default news chain (its news endpoints are paid; free
    # tier → 402), even though fmp_api_key is set here — the chain follows provider_news_priority.
    assert news == ["finnhub", "marketaux", "alpha_vantage", "newsdata", "google_news_rss"]
    assert topic == ["gdelt_doc", "guardian", "marketaux", "newsdata", "google_news_rss"]


def test_fmp_reactivates_when_added_back_to_chain() -> None:
    # The FmpProvider stays registered; a paid-plan user re-enables it purely via config, no code
    # change — putting `fmp` back in provider_news_priority makes it appear in the chain again.
    settings = Settings(
        _env_file=None,
        fmp_api_key="d",
        provider_news_priority="finnhub,fmp,google_news_rss",
    )
    reg = build_default_registry(settings)
    news = [p.name for p in reg._chain(settings.news_priority, NewsProvider)]  # type: ignore[type-abstract]
    assert news == ["fmp", "google_news_rss"]  # finnhub skipped (no key); fmp keyed → active


def test_topic_failover_skips_empty_provider() -> None:
    # GDELT returns 200-with-zero-articles (no error) → chain must fall through to Guardian.
    reg = _topic_registry(_FakeTopic("gdelt_doc", _bundle(0)), _FakeTopic("guardian", _bundle(3)))
    out = reg.get_topic_news(["ai memory"], date(2026, 5, 1), date(2026, 6, 2))
    assert len(out) == 3


def test_topic_failover_skips_errored_provider() -> None:
    # GDELT 429s → chain falls through to Guardian.
    reg = _topic_registry(
        _FakeTopic("gdelt_doc", error=ProviderRateLimit("gdelt_doc", "429")),
        _FakeTopic("guardian", _bundle(2)),
    )
    assert len(reg.get_topic_news(["x"], date(2026, 5, 1), date(2026, 6, 2))) == 2


def test_topic_all_empty_returns_empty_not_error() -> None:
    reg = _topic_registry(_FakeTopic("gdelt_doc", _bundle(0)), _FakeTopic("guardian", _bundle(0)))
    assert len(reg.get_topic_news(["x"], date(2026, 5, 1), date(2026, 6, 2))) == 0
