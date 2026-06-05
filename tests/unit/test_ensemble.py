"""Tests for the ensemble forecaster (linear probability pool)."""

from __future__ import annotations

from datetime import date

import numpy as np

from stock_agent.forecasting.ensemble import EnsembleForecast, blend_forecasts
from stock_agent.forecasting.historical import sample_to_forecast
from stock_agent.schemas.forecast import ScenarioForecast

_AS_OF = date(2024, 6, 1)


def _f(x: float | None) -> float:
    assert x is not None
    return x


def _member(seed: int, *, loc: float = 0.0, scale: float = 0.08) -> ScenarioForecast:
    """A consistent member forecast from a seeded return sample (buckets ↔ quantiles agree)."""
    rng = np.random.default_rng(seed)
    sample = rng.normal(loc, scale, size=4000)
    return sample_to_forecast(
        sample, ticker="NVDA", as_of=_AS_OF, horizon_days=20, model_name=f"m{seed}"
    )


def test_bucket_probs_and_linear_stats_are_weighted_average() -> None:
    f1, f2 = _member(1, loc=0.01), _member(2, loc=-0.01)
    out = blend_forecasts([f1, f2], ticker="NVDA", as_of=_AS_OF, horizon_days=20)
    # Equal-weight: each bucket prob is the average of the members'.
    for i, b in enumerate(out.buckets):
        avg = (f1.buckets[i].probability + f2.buckets[i].probability) / 2
        assert abs(b.probability - avg) < 1e-12
    assert abs(sum(b.probability for b in out.buckets) - 1.0) < 1e-9
    # Linear functionals are exact weighted means.
    assert abs(out.expected_return - (f1.expected_return + f2.expected_return) / 2) < 1e-12
    assert abs(out.upside_prob - (f1.upside_prob + f2.upside_prob) / 2) < 1e-12
    assert abs(out.downside_prob - (f1.downside_prob + f2.downside_prob) / 2) < 1e-12


def test_identical_members_blend_to_identity() -> None:
    f = _member(7)
    out = blend_forecasts([f, f, f], ticker="NVDA", as_of=_AS_OF, horizon_days=20)
    assert abs(out.expected_return - f.expected_return) < 1e-12
    for a, b in zip(out.buckets, f.buckets, strict=True):
        assert abs(a.probability - b.probability) < 1e-12
    # Quantiles reconstructed from the (identical) mixture CDF recover the member's.
    assert abs(_f(out.var_95) - _f(f.var_95)) < 5e-3
    assert abs(_f(out.var_99) - _f(f.var_99)) < 5e-3
    assert abs(_f(out.ci_high) - _f(f.ci_high)) < 5e-3


def test_quantiles_are_mixture_not_averaged_and_monotone() -> None:
    f1, f2 = _member(1, scale=0.05), _member(2, scale=0.15)  # very different spreads
    out = blend_forecasts([f1, f2], ticker="NVDA", as_of=_AS_OF, horizon_days=20)
    # Mixture quantile lies within the members' hull (a real CDF mix, not nonsense)…
    lo, hi = min(_f(f1.var_95), _f(f2.var_95)), max(_f(f1.var_95), _f(f2.var_95))
    assert lo - 1e-9 <= _f(out.var_95) <= hi + 1e-9
    # …and the quantiles stay ordered.
    assert _f(out.var_99) <= _f(out.var_95) <= _f(out.ci_high)


def test_weights_are_normalized() -> None:
    f1, f2 = _member(1), _member(2)
    a = blend_forecasts([f1, f2], [1.0, 1.0], ticker="NVDA", as_of=_AS_OF, horizon_days=20)
    b = blend_forecasts([f1, f2], [5.0, 5.0], ticker="NVDA", as_of=_AS_OF, horizon_days=20)
    assert abs(a.expected_return - b.expected_return) < 1e-12
    # Unequal weights shift toward the heavier member.
    c = blend_forecasts([f1, f2], [3.0, 1.0], ticker="NVDA", as_of=_AS_OF, horizon_days=20)
    assert abs(c.expected_return - (0.75 * f1.expected_return + 0.25 * f2.expected_return)) < 1e-12


class _Fake:
    def __init__(self, name: str, fc: ScenarioForecast) -> None:
        self.name, self._fc = name, fc

    def forecast(self, series, *, horizon_days, as_of=None):  # type: ignore[no-untyped-def]
        return self._fc


class _Raises:
    name = "boom"

    def forecast(self, series, *, horizon_days, as_of=None):  # type: ignore[no-untyped-def]
        raise ValueError("member exploded")


class _Series:
    ticker = "NVDA"

    class _Bar:
        date = _AS_OF

    bars = [_Bar()]


def test_fallback_member_is_dropped() -> None:
    good = _member(1)
    fell_back = _member(2).model_copy(update={"notes": "… Fell back to historical_sim."})
    ens = EnsembleForecast([_Fake("good", good), _Fake("ml_logistic", fell_back)])
    out = ens.forecast(_Series(), horizon_days=20)  # type: ignore[arg-type]
    # Only the good member survives → blend of one == that member.
    assert abs(out.expected_return - good.expected_return) < 1e-12
    assert "good" in (out.notes or "")


def test_member_exception_is_isolated() -> None:
    good = _member(3)
    ens = EnsembleForecast([_Raises(), _Fake("good", good)])
    out = ens.forecast(_Series(), horizon_days=20)  # type: ignore[arg-type]
    assert abs(out.expected_return - good.expected_return) < 1e-12
