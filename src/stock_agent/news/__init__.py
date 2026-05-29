"""News pipeline: fetch -> clean -> dedup -> rank (no LLM)."""

from stock_agent.news.clean import canonical_url, clean_article, clean_articles
from stock_agent.news.dedup import deduplicate
from stock_agent.news.fetch import NewsFetcher
from stock_agent.news.rank import rank_articles

__all__ = [
    "canonical_url",
    "clean_article",
    "clean_articles",
    "deduplicate",
    "rank_articles",
    "NewsFetcher",
]
