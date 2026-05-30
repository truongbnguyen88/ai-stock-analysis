"""Synthesizer (Role C) tests with a fake LLM — grounding is the key guard."""

from __future__ import annotations

import json
from datetime import date

import pytest

from stock_agent.indicators.snapshot import IndicatorSnapshot
from stock_agent.llm.synthesizer import SynthesisGuardError, synthesize
from stock_agent.schemas.earnings import EarningsContext
from stock_agent.schemas.forecast import ProbBucket, ScenarioForecast
from stock_agent.schemas.synthesis import Synthesis


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def complete_json(self, *, system: str, user: str, max_tokens: int = 2048) -> str:
        out = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return out


def _forecast() -> ScenarioForecast:
    buckets = [
        ProbBucket(label="< -10%", lower=None, upper=-0.10, probability=0.1),
        ProbBucket(label="-10% to -5%", lower=-0.10, upper=-0.05, probability=0.1),
        ProbBucket(label="-5% to 0%", lower=-0.05, upper=0.0, probability=0.12),
        ProbBucket(label="0% to +5%", lower=0.0, upper=0.05, probability=0.3),
        ProbBucket(label="+5% to +10%", lower=0.05, upper=0.10, probability=0.2),
        ProbBucket(label="> +10%", lower=0.10, upper=None, probability=0.18),
    ]
    return ScenarioForecast(
        ticker="NVDA",
        as_of=date(2025, 1, 31),
        horizon_days=20,
        model_name="ml_xgboost",
        buckets=buckets,
        expected_return=0.045,
        upside_prob=0.68,
        downside_prob=0.32,
        var_95=-0.08,
    )


def _snapshot() -> IndicatorSnapshot:
    return IndicatorSnapshot(
        as_of=date(2025, 1, 31),
        last_close=200.0,
        rsi14=72.0,
        hist_vol_annualized=0.40,
        macd_hist=1.5,
        trend_label="uptrend",
    )


def _earnings() -> EarningsContext:
    return EarningsContext(
        ticker="NVDA",
        as_of=date(2025, 1, 31),
        next_earnings_date=date(2025, 2, 8),
        days_to_next_earnings=8,
        horizon_days=20,
        earnings_in_horizon=True,
    )


def _payload(overview: str) -> str:
    return json.dumps(
        {
            "overview": overview,
            "alignments": ["Uptrend and positive MACD support the model's upside tilt."],
            "tensions": ["Negative news sentiment and earnings in 8 days temper the 68% upside."],
            "confidence": "Moderate — earnings inside the window widen the real distribution.",
        }
    )


_SENTIMENT = {
    "avg_sentiment": -0.30,
    "sentiment_coverage": 0.4,
    "pct_positive": 0.2,
    "pct_negative": 0.5,
}


def _run(llm: FakeLLM) -> Synthesis:
    return synthesize(
        "NVDA",
        llm=llm,
        forecasts=[_forecast()],
        snapshot=_snapshot(),
        news_sentiment=_SENTIMENT,
        earnings=_earnings(),
    )


def test_synthesis_grounded_numbers_pass() -> None:
    # Cites 68% (from forecast), -0.30 (sentiment), 8 days (earnings) — all grounded.
    out = _run(FakeLLM([_payload("The model's 68% upside meets -0.30 sentiment.")]))
    assert "68%" in out.overview
    assert out.tensions and out.alignments


def test_synthesis_invented_number_triggers_retry_then_succeeds() -> None:
    bad = _payload("I'd revise this to a 55% chance given the news.")  # 55% is not in inputs
    good = _payload("The model's 68% upside is tempered by negative sentiment.")
    llm = FakeLLM([bad, good])
    out = _run(llm)
    assert llm.calls == 2  # retried once
    assert "55%" not in out.overview


def test_synthesis_persistent_invention_raises() -> None:
    bad = _payload("Really it's a 55.5% chance, not what the model says.")
    with pytest.raises(SynthesisGuardError):
        _run(FakeLLM([bad, bad]))
