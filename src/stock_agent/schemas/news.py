"""News domain models — normalized article shape shared across news providers."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl


class Article(BaseModel):
    """A single news article, normalized across providers.

    ``sentiment`` and ``relevance`` are OPTIONAL numeric fields populated by our
    own models/heuristics (or provider-supplied scores) — never by the LLM, per
    the numbers-vs-narrative invariant.
    """

    title: str
    url: HttpUrl
    source: str
    published_at: datetime
    summary: str | None = None
    sentiment: float | None = None  # model/provider score, conventionally [-1, 1]
    relevance: float | None = None  # ranking score, [0, 1]
    topics: list[str] = Field(default_factory=list)


class NewsBundle(BaseModel):
    """A collection of articles for one ticker over a fetch window."""

    ticker: str
    articles: list[Article] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.articles)
