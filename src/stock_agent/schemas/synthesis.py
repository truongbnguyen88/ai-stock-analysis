"""Integrated-analysis (Role C) domain model.

The LLM-authored reconciliation of the quantitative forecast with the qualitative
signals. Contains no invented numbers (enforced by the numeric-grounding guard).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Synthesis(BaseModel):
    """An integrated read tying the forecast numbers to the news/earnings/technicals."""

    overview: str = ""
    alignments: list[str] = Field(default_factory=list)  # where quant & qualitative agree
    tensions: list[str] = Field(default_factory=list)  # where they disagree (the key insight)
    confidence: str = ""  # how much to trust the forecast, and why
