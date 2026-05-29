"""Shared pydantic domain models — the typed contracts between all layers."""

from stock_agent.schemas.forecast import CalibrationStatus, ProbBucket, ScenarioForecast
from stock_agent.schemas.market import Fundamentals, PriceBar, PriceSeries
from stock_agent.schemas.news import Article, NewsBundle
from stock_agent.schemas.report import (
    Citation,
    NewsAnalysisSection,
    ResearchReport,
    TechnicalAnalysisSection,
)

__all__ = [
    # market
    "PriceBar",
    "PriceSeries",
    "Fundamentals",
    # news
    "Article",
    "NewsBundle",
    # forecast
    "ProbBucket",
    "ScenarioForecast",
    "CalibrationStatus",
    # report
    "ResearchReport",
    "Citation",
    "NewsAnalysisSection",
    "TechnicalAnalysisSection",
]
