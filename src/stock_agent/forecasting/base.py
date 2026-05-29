"""Forecast model interface.

All forecasters (historical baseline now; Monte-Carlo and ML in Phase 5) share
this structural interface so they are swappable and directly comparable in
backtests. Every model returns a model-generated ``ScenarioForecast`` — the LLM
never produces these.
"""

from __future__ import annotations

from datetime import date as Date
from typing import Protocol, runtime_checkable

from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceSeries


@runtime_checkable
class ForecastModel(Protocol):
    """Produces a forward-return scenario distribution for a price series."""

    name: str

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        """Return the scenario forecast for ``horizon_days`` ahead of ``as_of``."""
        ...
