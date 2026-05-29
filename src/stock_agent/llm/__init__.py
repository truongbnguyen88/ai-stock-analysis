"""LLM layer (Role A): provider-agnostic client, guards, and news summarizer.

The LLM summarizes and explains news. It never produces probabilities, forecasts,
or recommendations — enforced by guards in ``guards`` and ``news_summarizer``.
"""

from stock_agent.llm.client import AnthropicClient, LLMError, TextLLM
from stock_agent.llm.guards import (
    CitedPoint,
    NewsSummary,
    find_forecast_violations,
    find_invalid_citations,
    sanitize_citations,
)
from stock_agent.llm.news_summarizer import SummaryGuardError, summarize_news

__all__ = [
    "AnthropicClient",
    "TextLLM",
    "LLMError",
    "NewsSummary",
    "CitedPoint",
    "find_forecast_violations",
    "find_invalid_citations",
    "sanitize_citations",
    "summarize_news",
    "SummaryGuardError",
]
