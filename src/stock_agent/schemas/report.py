"""Research-report domain model — the final assembled artifact.

DESIGN INVARIANT: there is intentionally NO recommendation / buy-sell field. The
tool is non-advisory by construction (see docs/ARCHITECTURE.md §1). Narrative
fields are LLM-authored; quantitative fields (forecasts, fundamentals, indicator
snapshots) come from deterministic/model code.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field, HttpUrl

from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import Fundamentals


class Citation(BaseModel):
    """A source reference backing a claim in the report."""

    title: str
    url: HttpUrl
    source: str


class TechnicalAnalysisSection(BaseModel):
    """Deterministic technical findings.

    ``indicators`` holds the raw computed snapshot (numbers come from code);
    ``narrative`` is an LLM explanation of those numbers.
    """

    narrative: str = ""
    indicators: dict[str, float] = Field(default_factory=dict)


class NewsAnalysisSection(BaseModel):
    """LLM-authored qualitative news synthesis (no numbers invented by the LLM)."""

    recent_developments: list[str] = Field(default_factory=list)
    positive_themes: list[str] = Field(default_factory=list)
    negative_themes: list[str] = Field(default_factory=list)
    emerging_risks: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The complete research report assembled by ``reports/builder.py``."""

    ticker: str
    as_of: Date

    # Executive summary
    executive_summary: str = ""
    key_observations: list[str] = Field(default_factory=list)
    confidence_notes: str = ""

    # Body sections
    technical_analysis: TechnicalAnalysisSection = Field(default_factory=TechnicalAnalysisSection)
    news_analysis: NewsAnalysisSection = Field(default_factory=NewsAnalysisSection)
    fundamentals: Fundamentals | None = None
    forecasts: list[ScenarioForecast] = Field(default_factory=list)

    # Structured signal lists
    bullish_factors: list[str] = Field(default_factory=list)
    bearish_factors: list[str] = Field(default_factory=list)
    risk_flags: list[str] = Field(default_factory=list)

    # Honesty section — missing data, sparse news, conflicting signals, model limits
    uncertainty_notes: list[str] = Field(default_factory=list)

    # Provenance
    citations: list[Citation] = Field(default_factory=list)
