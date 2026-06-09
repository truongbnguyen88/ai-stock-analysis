"""News fetch orchestration: registry -> clean -> dedup -> rank.

Thin layer mirroring ``data.loader.PriceLoader``. Produces a ranked, deduplicated
``NewsBundle`` ready for the LLM summarizer (Step 10) or report assembly.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date as Date
from datetime import timedelta

from stock_agent.llm.client import TextLLM
from stock_agent.logging_config import get_logger
from stock_agent.news.clean import clean_articles
from stock_agent.news.dedup import deduplicate
from stock_agent.news.rank import Order, rank_articles
from stock_agent.news.topic_expand import expand_topic_keywords
from stock_agent.news.topics import ResolvedTopic, gdelt_query_expression, resolve_topic
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.news import NewsBundle

log = get_logger(__name__)


class NewsFetcher:
    """Fetches and prepares company news via the provider registry."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def fetch(
        self,
        ticker: str,
        lookback_days: int = 30,
        *,
        company_name: str | None = None,
        top_n: int | None = 25,
        end: Date | None = None,
        order: Order = "relevance",
    ) -> NewsBundle:
        """Return a cleaned, deduplicated, ranked ``NewsBundle`` (see ``rank_articles`` order)."""
        end = end or Date.today()
        start = end - timedelta(days=lookback_days)

        raw = self._registry.get_company_news(ticker, start, end)
        cleaned = clean_articles(raw.articles)
        deduped = deduplicate(cleaned)
        ranked = rank_articles(
            deduped,
            ticker,
            company_name=company_name,
            lookback_days=lookback_days,
            top_n=top_n,
            order=order,
        )
        log.info(
            "news.fetch",
            ticker=ticker,
            raw=len(raw.articles),
            cleaned=len(cleaned),
            deduped=len(deduped),
            ranked=len(ranked),
        )
        return NewsBundle(ticker=ticker, articles=ranked)


class TopicNewsFetcher:
    """Fetches and prepares theme/keyword news via the topic-provider chain (Enhancement C).

    Mirrors ``NewsFetcher`` but for theme-scoped queries: resolves a theme name to a
    query spec (registry or free-form), fetches via ``registry.get_topic_news``, then
    reuses the same clean -> dedup -> rank pipeline (newest-first by default).
    """

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def fetch(
        self,
        topic: str,
        lookback_days: int = 14,
        *,
        top_n: int | None = 25,
        end: Date | None = None,
        order: Order = "recency",
        llm: TextLLM | None = None,
        expand: bool = False,
    ) -> tuple[ResolvedTopic, NewsBundle]:
        """Return the resolved topic and a cleaned/deduped/ranked ``NewsBundle``.

        The ``ResolvedTopic`` is returned alongside so callers can surface the
        matched keywords/query for transparency ("what 'robotics' matched").

        ``expand`` + ``llm``: for a FREE-FORM topic (not a curated registry theme),
        widen the single phrase into OR-able search keywords via the LLM, improving
        recall. Curated themes are already precise and left untouched. Best-effort —
        expansion failure falls back to the original phrase.
        """
        resolved = resolve_topic(topic)
        if expand and llm is not None and not resolved.known:
            resolved = replace(resolved, keywords=expand_topic_keywords(topic, llm))
        query = gdelt_query_expression(resolved)
        end = end or Date.today()
        start = end - timedelta(days=lookback_days)

        # Over-fetch a bit so dedup/ranking has material; cap the returned set to top_n.
        fetch_n = (top_n or 25) * 2
        raw = self._registry.get_topic_news(resolved.keywords, start, end, top_n=fetch_n)
        cleaned = clean_articles(raw.articles)
        deduped = deduplicate(cleaned)
        # Rank by recency against the topic label (mention scoring is moot for themes).
        ranked = rank_articles(
            deduped, resolved.label, lookback_days=lookback_days, top_n=top_n, order=order
        )
        log.info(
            "topic_news.fetch",
            topic=resolved.topic,
            known=resolved.known,
            query=query,
            raw=len(raw.articles),
            ranked=len(ranked),
        )
        return resolved, NewsBundle(ticker=resolved.label, articles=ranked)
