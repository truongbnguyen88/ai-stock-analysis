"""Provider registry — capability resolution and fallback orchestration.

Given a set of instantiated providers and the priority order from settings, the
registry resolves a fallback *chain* per capability and drives it:

- prices:    failover — try providers in order, return the first success.
- news:      merge   — combine articles from all available providers (dedup is a
             later pipeline step, Phase 3), with failover semantics if all fail.

Providers that are absent, lack the capability, or report ``available() is False``
(e.g. missing API key) are silently skipped, so configuration alone controls
which providers participate.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date
from typing import Literal, TypeVar

from stock_agent.logging_config import get_logger
from stock_agent.providers.base import (
    EarningsProvider,
    NewsProvider,
    PriceProvider,
    Provider,
    ProviderError,
    ProviderUnavailable,
    TopicNewsProvider,
)
from stock_agent.schemas.market import PriceSeries
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.settings import Settings

log = get_logger(__name__)

NewsMode = Literal["merge", "failover"]

# Bound to Provider so resolved chains expose name/available; isinstance narrows
# to the specific capability (PriceProvider/NewsProvider) within each method.
P = TypeVar("P", bound=Provider)


class ProviderRegistry:
    """Holds providers and resolves/drives capability fallback chains."""

    def __init__(self, providers: Sequence[Provider], settings: Settings) -> None:
        self._by_name: dict[str, Provider] = {p.name: p for p in providers}
        self._settings = settings

    def _chain(self, names: Sequence[str], protocol: type[P]) -> list[P]:
        """Resolve the ordered, usable providers for a capability.

        Filters the configured priority list to providers that exist, structurally
        satisfy ``protocol``, and are currently available.
        """
        chain: list[P] = []
        for name in names:
            provider = self._by_name.get(name)
            if provider is None:
                continue
            if not isinstance(provider, protocol):  # narrows to the capability type
                continue
            if not provider.available():
                log.debug("provider.skip_unavailable", provider=name)
                continue
            chain.append(provider)
        return chain

    # ---- prices: failover ----------------------------------------------------
    def get_daily_prices(self, ticker: str, start: Date, end: Date) -> PriceSeries:
        """Return prices from the first provider in the chain that succeeds."""
        # Passing a runtime_checkable Protocol class as type[P] is intentional;
        # mypy's type-abstract check flags it, but isinstance handles it fine.
        chain = self._chain(self._settings.price_priority, PriceProvider)  # type: ignore[type-abstract]
        if not chain:
            raise ProviderUnavailable("registry", "no price providers available")

        last_error: ProviderError | None = None
        for provider in chain:
            try:
                series = provider.get_daily_prices(ticker, start, end)
                log.info(
                    "provider.prices_ok", provider=provider.name, ticker=ticker, bars=len(series)
                )
                return series
            except ProviderError as exc:
                log.warning(
                    "provider.prices_failed", provider=provider.name, ticker=ticker, error=str(exc)
                )
                last_error = exc
                continue
        # All providers failed; surface the last concrete error.
        assert last_error is not None
        raise last_error

    # ---- news: merge (default) or failover -----------------------------------
    def get_company_news(
        self, ticker: str, start: Date, end: Date, *, mode: NewsMode = "merge"
    ) -> NewsBundle:
        """Return company news.

        ``merge`` (default) concatenates articles from every available provider —
        broader coverage; the dedup step (Phase 3) removes overlap later.
        ``failover`` returns the first provider's results only.
        """
        chain = self._chain(self._settings.news_priority, NewsProvider)  # type: ignore[type-abstract]
        if not chain:
            raise ProviderUnavailable("registry", "no news providers available")

        collected: list[Article] = []
        last_error: ProviderError | None = None
        any_success = False
        for provider in chain:
            try:
                bundle = provider.get_company_news(ticker, start, end)
                any_success = True
                collected.extend(bundle.articles)
                log.info("provider.news_ok", provider=provider.name, ticker=ticker, n=len(bundle))
                if mode == "failover":
                    break
            except ProviderError as exc:
                log.warning(
                    "provider.news_failed", provider=provider.name, ticker=ticker, error=str(exc)
                )
                last_error = exc
                continue

        if not any_success:
            assert last_error is not None
            raise last_error
        return NewsBundle(ticker=ticker, articles=collected)

    # ---- topic news: failover -------------------------------------------------
    def get_topic_news(
        self, keywords: Sequence[str], start: Date, end: Date, *, top_n: int = 25
    ) -> NewsBundle:
        """Return theme/keyword news from the first topic provider that succeeds.

        Failover (not merge): topic providers are heterogeneous (GDELT DOC,
        Marketaux search) and each builds its own native query from ``keywords``,
        so we take the first usable source rather than blending query semantics.
        """
        chain = self._chain(self._settings.topic_priority, TopicNewsProvider)  # type: ignore[type-abstract]
        if not chain:
            raise ProviderUnavailable("registry", "no topic-news providers available")
        last_error: ProviderError | None = None
        last_empty: NewsBundle | None = None
        for provider in chain:
            try:
                bundle = provider.get_topic_news(keywords, start, end, top_n=top_n)
            except ProviderError as exc:
                log.warning(
                    "provider.topic_news_failed",
                    provider=provider.name, keywords=list(keywords), error=str(exc),
                )
                last_error = exc
                continue
            if len(bundle) > 0:
                log.info(
                    "provider.topic_news_ok",
                    provider=provider.name, n=len(bundle), keywords=list(keywords),
                )
                return bundle
            # A 200-with-zero-articles is NOT a stop signal — a flaky source (e.g. GDELT) often
            # returns empty without erroring; fall through so a healthier provider can answer.
            log.info("provider.topic_news_empty", provider=provider.name, keywords=list(keywords))
            last_empty = bundle
        if last_empty is not None:  # every provider was reachable but found nothing — honest empty
            return last_empty
        assert last_error is not None  # every provider errored
        raise last_error

    def get_topic_tone(
        self, keywords: Sequence[str], start: Date, end: Date
    ) -> dict[str, object] | None:
        """Aggregate topic sentiment from the first topic provider that supports it.

        ``get_topic_tone`` is an OPTIONAL capability (only GDELT exposes it today),
        so we probe via ``getattr`` rather than the Protocol. Returns ``None`` when
        no provider supports it or all fail — tone is best-effort context, never fatal.
        """
        chain = self._chain(self._settings.topic_priority, TopicNewsProvider)  # type: ignore[type-abstract]
        for provider in chain:
            fn = getattr(provider, "get_topic_tone", None)
            if not callable(fn):
                continue
            try:
                tone: dict[str, object] | None = fn(keywords, start, end)
                return tone
            except ProviderError as exc:
                log.warning("provider.topic_tone_failed", provider=provider.name, error=str(exc))
                continue
        return None

    # ---- earnings: failover ---------------------------------------------------
    def get_earnings_dates(self, ticker: str) -> list[Date]:
        """Return earnings announcement dates from the first provider that succeeds."""
        chain = self._chain(self._settings.earnings_priority, EarningsProvider)  # type: ignore[type-abstract]
        if not chain:
            raise ProviderUnavailable("registry", "no earnings providers available")
        last_error: ProviderError | None = None
        for provider in chain:
            try:
                return provider.get_earnings_dates(ticker)
            except ProviderError as exc:
                log.warning(
                    "provider.earnings_failed",
                    provider=provider.name,
                    ticker=ticker,
                    error=str(exc),
                )
                last_error = exc
                continue
        assert last_error is not None
        raise last_error

    def close(self) -> None:
        """Close any providers that hold resources (e.g. HTTP clients)."""
        for provider in self._by_name.values():
            closer = getattr(provider, "close", None)
            if callable(closer):
                closer()


def build_default_registry(settings: Settings) -> ProviderRegistry:
    """Construct the standard provider set wired to a shared disk cache.

    Imports concrete providers lazily so the registry core stays decoupled from
    any specific provider module.
    """
    from stock_agent.providers._cache import DiskCache
    from stock_agent.providers.alpha_vantage import AlphaVantageProvider
    from stock_agent.providers.finnhub import FinnhubProvider
    from stock_agent.providers.fmp import FmpProvider
    from stock_agent.providers.gdelt_doc import GdeltDocProvider
    from stock_agent.providers.google_news_rss import GoogleNewsRssProvider
    from stock_agent.providers.guardian import GuardianProvider
    from stock_agent.providers.marketaux import MarketauxProvider
    from stock_agent.providers.newsdata import NewsDataProvider
    from stock_agent.providers.thenewsapi import TheNewsApiProvider
    from stock_agent.providers.tiingo import TiingoProvider
    from stock_agent.providers.yfinance_provider import YFinanceProvider

    cache = DiskCache(settings.cache_dir, settings.cache_ttl_seconds)
    # Providers whose key is absent are skipped at chain-resolution time (available()==False), so
    # registering them here is harmless — they activate only once their key is set.
    providers: list[Provider] = [
        YFinanceProvider(cache=cache),
        FinnhubProvider(settings=settings, cache=cache),
        MarketauxProvider(settings=settings, cache=cache),
        AlphaVantageProvider(settings=settings, cache=cache),
        FmpProvider(settings=settings, cache=cache),  # per-ticker finance news (paid; off default)
        TiingoProvider(settings=settings, cache=cache),  # per-ticker news (keyed, 1k/day free)
        GdeltDocProvider(settings=settings, cache=cache),  # keyless topic/theme news
        GuardianProvider(settings=settings, cache=cache),  # topic news (keyed, 5k/day)
        NewsDataProvider(settings=settings, cache=cache),  # per-ticker + topic (keyed)
        TheNewsApiProvider(settings=settings, cache=cache),  # topic news (keyed, 100/day)
        GoogleNewsRssProvider(settings=settings, cache=cache),  # per-ticker + topic (KEYLESS)
    ]
    return ProviderRegistry(providers, settings)
