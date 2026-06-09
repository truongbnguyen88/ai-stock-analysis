"""conditional_outlook agent tool (#7) — offline via FakeProvider price series."""

from __future__ import annotations

from datetime import date, timedelta

from stock_agent.agent.tools import ToolExecutor
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.conditional import ConditionalStudy
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.settings import Settings


def _series(n: int = 160) -> PriceSeries:
    # Drifting price with periodic +7% jumps so driver shocks (and events) occur.
    bars = []
    price = 100.0
    for i in range(n):
        price *= 1.07 if i % 25 == 0 and i > 0 else 1.002
        bars.append(
            PriceBar(date=date(2020, 1, 1) + timedelta(days=i), open=price, high=price,
                     low=price, close=price)
        )
    return PriceSeries(ticker="X", bars=bars)


def _executor() -> ToolExecutor:
    # FakeProvider returns the same series for any symbol, so target and driver
    # share a calendar — fine for exercising the tool plumbing + schema.
    fake = FakeProvider("fake", prices=_series())
    registry = ProviderRegistry(
        [fake], Settings(_env_file=None, provider_price_priority="fake")
    )
    return ToolExecutor(Settings(_env_file=None), registry=registry)


def test_conditional_outlook_returns_valid_study() -> None:
    r = _executor().execute(
        "conditional_outlook",
        {"target": "DAL", "driver": "USO", "shock_pct": 5, "event_window_days": 5,
         "horizon_days": 10},
    )
    assert "error" not in r
    study = ConditionalStudy.model_validate(r)  # schema conformance
    assert study.target == "DAL" and study.driver == "USO"
    assert study.baseline.n > 0
    assert study.horizon_days == 10
    assert "not a forecast" in study.methodology.lower()


def test_conditional_outlook_rejects_bad_direction() -> None:
    r = _executor().execute(
        "conditional_outlook", {"target": "DAL", "driver": "USO", "direction": "sideways"}
    )
    assert "error" in r and "direction" in r["error"]


def test_conditional_outlook_bounds_shock_and_horizon() -> None:
    ex = _executor()
    assert "error" in ex.execute(
        "conditional_outlook", {"target": "DAL", "driver": "USO", "shock_pct": 99}
    )
    assert "error" in ex.execute(
        "conditional_outlook", {"target": "DAL", "driver": "USO", "horizon_days": 999}
    )
