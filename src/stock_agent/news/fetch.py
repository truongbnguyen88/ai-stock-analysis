"""News fetch orchestration: registry -> clean -> dedup -> rank.

Thin layer mirroring ``data.loader.PriceLoader``. Produces a ranked, deduplicated
``NewsBundle`` ready for the LLM summarizer (Step 10) or report assembly.
"""

from __future__ import annotations

from datetime import date as Date
from datetime import timedelta

from stock_agent.logging_config import get_logger
from stock_agent.news.clean import clean_articles
from stock_agent.news.dedup import deduplicate
from stock_agent.news.rank import rank_articles
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
    ) -> NewsBundle:
        """Return a cleaned, deduplicated, relevance-ranked ``NewsBundle``."""
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
