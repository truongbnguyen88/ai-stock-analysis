"""Research-answer domain models (RAG P7).

A ``GroundedAnswer`` is the output of the single SEC-grounded synthesis call: a cited
answer whose every source marker resolves to a chunk that was actually retrieved (the
citation guard enforces this). ``insufficient_evidence`` is the honest-refusal signal —
when retrieval found nothing relevant, the answer is "Insufficient evidence found." and
no claim is made.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field

from stock_agent.schemas.forecast import ScenarioForecast


class SourceCitation(BaseModel):
    """Resolves an inline ``[n]`` marker in the answer to the chunk it cites."""

    marker: int  # the [n] used inline in the answer text
    chunk_id: str  # the retrieved chunk this marker refers to (⊆ the EvidenceSet)
    label: str  # human-readable, e.g. "NVDA 10-K Feb 26, 2025 — Item 1A. Risk Factors"


class GroundedAnswer(BaseModel):
    """A SEC-grounded answer to one question, with resolved citations."""

    question: str
    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)
    insufficient_evidence: bool = False


class NewsAnalysis(BaseModel):
    """Structured recent-news analysis surfaced in the brief.

    Strings only (the qualitative points flattened from the LLM ``NewsSummary``) so this
    pure schema carries no ``llm/`` dependency. Sentiment shares are provider scores
    aggregated by ``features.news_features`` (no LLM); ``None`` when no article carried a
    score. All fields are descriptive — never a recommendation (DESIGN INVARIANT §2).
    """

    lookback_days: int = 0  # calendar-day window the articles were pulled over
    article_count: int = 0
    overview: str = ""
    key_themes: list[str] = Field(default_factory=list)
    bullish: list[str] = Field(default_factory=list)
    bearish: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    # Sentiment composition (fraction of scored articles). None => no scores available.
    pct_positive: float | None = None
    pct_negative: float | None = None
    sentiment_coverage: float | None = None  # fraction of articles that carried a score


class ResearchMemo(BaseModel):
    """An integrated research memo for one ticker (RAG P8).

    Combines QUANTITATIVE sections rendered directly from the models — ``technical_indicators``
    and ``forecasts`` (the LLM never produces these numbers) — with QUALITATIVE narrative
    synthesized in one grounded call. Every claim drawn from a filing carries an inline ``[n]``
    marker resolved in ``citations`` (citation guard); narrative figures must trace to the inputs
    (number-grounding guard). DESIGN INVARIANT: no recommendation / buy-sell field.
    """

    ticker: str
    as_of: Date

    # Quantitative — copied verbatim from the modules.
    technical_indicators: dict[str, float] = Field(default_factory=dict)
    forecasts: list[ScenarioForecast] = Field(default_factory=list)

    # Qualitative narrative — produced by the single synthesis call (SEC claims cited [n]).
    executive_summary: str = ""
    management_commentary: str = ""
    business_drivers: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    bullish_evidence: list[str] = Field(default_factory=list)
    bearish_evidence: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    recent_news: list[str] = Field(default_factory=list)  # news themes (from the summary, if any)
    news: NewsAnalysis | None = None  # full recent-news analysis (themes + insights + sentiment)

    citations: list[SourceCitation] = Field(default_factory=list)  # resolved SEC sources
