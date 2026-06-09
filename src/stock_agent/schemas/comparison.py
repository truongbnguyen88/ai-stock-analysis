"""Multi-ticker comparison domain models (Enhancement B).

Structured, comparable shapes for the batch agent tools (``compare_forecasts`` /
``compare_news``). Every numeric field is populated from the existing
``forecasting`` / news modules — never the LLM (numbers-vs-narrative invariant).
The LLM only narrates the cross-ticker comparison; it never fills these in.

A per-ticker row may instead carry an ``error`` (e.g. no data / fetch failure) so
one bad ticker never sinks the whole comparison — the agent reports it and moves on.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field


class HeadlineRef(BaseModel):
    """One newest-first headline reference for a ticker (for citation in the UI)."""

    title: str
    source: str
    published: Date
    url: str


class TickerForecastRow(BaseModel):
    """One ticker's forecast summary in a multi-ticker comparison.

    Numbers come from a single ``run_forecast`` call (see ``pipelines.forecast``);
    ``prob_large_move`` is derived from that forecast's buckets via the same
    ``large_move_breakdown`` the single-ticker tool uses. All returns are fractional.
    """

    ticker: str
    error: str | None = None  # set => the other fields are absent for this ticker

    horizon_days: int | None = None
    model_name: str | None = None
    expected_return: float | None = None  # fractional
    upside_prob: float | None = Field(default=None, ge=0.0, le=1.0)  # P(return > 0)
    downside_prob: float | None = Field(default=None, ge=0.0, le=1.0)  # P(return < 0)
    var_95: float | None = None  # fractional (typically negative)
    ci_low: float | None = None
    ci_high: float | None = None
    # Magnitude signal: P(|return| > k) at the horizon's default bucket edge.
    prob_large_move: float | None = Field(default=None, ge=0.0, le=1.0)
    large_move_threshold_pct: int | None = None
    notes: str | None = None  # e.g. ML fell back to the baseline


class ForecastComparison(BaseModel):
    """Side-by-side forecast comparison across several tickers at one horizon."""

    horizon_days: int = Field(gt=0)
    model: str
    rows: list[TickerForecastRow]
    skipped: list[str] = Field(default_factory=list)  # tickers dropped over the cap


class TickerNewsRow(BaseModel):
    """One ticker's news/sentiment summary in a multi-ticker comparison.

    Sentiment is numeric and comes from ``build_news_features`` (Alpha Vantage
    scores by default); headlines are the newest-first articles. No LLM numbers.
    """

    ticker: str
    error: str | None = None

    article_count: int | None = None
    avg_sentiment: float | None = None  # conventionally [-1, 1]
    pct_positive: float | None = Field(default=None, ge=0.0, le=1.0)
    pct_negative: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_source: str | None = None
    top_headlines: list[HeadlineRef] = Field(default_factory=list)


class NewsComparison(BaseModel):
    """Side-by-side news/sentiment comparison across several tickers."""

    days: int = Field(gt=0)
    rows: list[TickerNewsRow]
    skipped: list[str] = Field(default_factory=list)
