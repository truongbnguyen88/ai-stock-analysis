"""Tests for the offline conformal-calibration primitives (artifact, fit, scoring)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from stock_agent.forecasting.conformal_calibrate import (
    ConformalArtifact,
    ConformalEntry,
    fit_entry,
    interval_scores,
    new_artifact,
)
from stock_agent.schemas.forecast import ProbBucket, ScenarioForecast
from stock_agent.schemas.market import PriceBar, PriceSeries


def test_artifact_roundtrip_and_lookup(tmp_path: Path) -> None:
    art = new_artifact(0.90, date(2025, 1, 1))
    e = ConformalEntry(q=0.012, n=500, coverage_before=0.83, coverage_after=0.90)
    art.entries["ensemble"] = {20: e}
    p = tmp_path / "conformal.json"
    art.save(p)
    loaded = ConformalArtifact.load(p)
    assert loaded is not None
    assert loaded.ci_level == 0.90
    q = loaded.q_for("ensemble", 20)
    assert q is not None and abs(q - 0.012) < 1e-12
    assert loaded.q_for("ensemble", 60) is None  # not calibrated
    assert loaded.q_for("nope", 20) is None


def test_load_missing_is_none(tmp_path: Path) -> None:
    assert ConformalArtifact.load(tmp_path / "absent.json") is None


def test_fit_entry_corrects_undercoverage() -> None:
    # Narrow CI [-0.04, 0.04] but realized moves ±0.10 → under-covers → q>0 → fixed.
    import numpy as np

    ys = np.random.default_rng(0).normal(0, 0.10, size=400)
    triples = [(-0.04, 0.04, float(y)) for y in ys]
    e = fit_entry(triples, ci_level=0.90)
    assert e is not None
    assert e.coverage_before < 0.80
    assert e.q > 0
    assert e.coverage_after >= 0.88  # near the 90% target on the same pool


def test_fit_entry_empty_is_none() -> None:
    assert fit_entry([], ci_level=0.90) is None


def _bar(d: date, close: float) -> PriceBar:
    return PriceBar(date=d, open=close, high=close, low=close, close=close, volume=1000)


class _FixedModel:
    """Forecaster that always returns the same ±5% CI (for scoring tests)."""

    name = "fixed"

    def forecast(self, series: PriceSeries, *, horizon_days: int, as_of=None) -> ScenarioForecast:  # type: ignore[no-untyped-def]
        return ScenarioForecast(
            ticker=series.ticker, as_of=series.bars[-1].date, horizon_days=horizon_days,
            model_name="fixed",
            buckets=[ProbBucket(label="x", lower=None, upper=None, probability=1.0)],
            expected_return=0.0, upside_prob=0.5, downside_prob=0.5,
            ci_level=0.90, ci_low=-0.05, ci_high=0.05,
        )


def test_interval_scores_only_after_cutoff_and_point_in_time() -> None:
    start = date(2024, 1, 1)
    bars = [_bar(start + timedelta(days=i), 100.0 + i) for i in range(120)]
    series = PriceSeries(ticker="T", bars=bars)
    cutoff = bars[80].date  # only as-ofs after index 80 should be scored
    triples = interval_scores(_FixedModel(), series, horizon_days=5, cal_cutoff=cutoff, stride=1)
    assert len(triples) > 0
    # All triples carry the fixed interval; realized is a real forward return.
    assert all(lo == -0.05 and hi == 0.05 for lo, hi, _ in triples)
    # Stride subsamples.
    strided = interval_scores(_FixedModel(), series, 5, cal_cutoff=cutoff, stride=5)
    assert len(strided) < len(triples)
