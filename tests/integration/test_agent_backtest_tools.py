"""Phase 6.5: agent backtest/calibration tools (offline, no network).

Drives the real Phase 6 backtest harness through the executor over a synthetic
FakeProvider series, then checks the agent loop can answer a calibration
question with grounded numbers.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from stock_agent.agent.runtime import ToolResponse, ToolUse, run_agent
from stock_agent.agent.tools import ToolExecutor
from stock_agent.backtesting.calibration import calibration_label
from stock_agent.providers.fake import FakeProvider
from stock_agent.providers.registry import ProviderRegistry
from stock_agent.schemas.market import PriceBar, PriceSeries
from stock_agent.settings import Settings


def _series(n: int = 700) -> PriceSeries:
    """Oscillating-but-slightly-up series → varied OOS labels across thresholds."""
    bars = []
    price = 100.0
    for i in range(n):
        price *= 1.001 if i % 2 == 0 else 0.9994
        c = round(price, 4)
        bars.append(
            PriceBar(
                date=date(2024, 1, 1) + timedelta(days=i),
                open=c,
                high=round(c * 1.01, 4),
                low=round(c * 0.99, 4),
                close=c,
            )
        )
    return PriceSeries(ticker="TST", bars=bars)


def _executor(n_bars: int = 700) -> ToolExecutor:
    fake = FakeProvider("fake", prices=_series(n_bars))
    registry = ProviderRegistry([fake], Settings(_env_file=None, provider_price_priority="fake"))
    return ToolExecutor(Settings(_env_file=None), registry=registry)


def _final(text: str) -> ToolResponse:
    return ToolResponse(text=text, tool_uses=[], stop_reason="end_turn", assistant_content=[])


# ---- calibration_label thresholds ---------------------------------------------
def test_calibration_label_buckets() -> None:
    assert calibration_label(0.02) == "well-calibrated"
    assert calibration_label(0.07) == "moderately calibrated"
    assert calibration_label(0.2) == "poorly calibrated"


# ---- run_backtest tool --------------------------------------------------------
def test_run_backtest_tool_shape() -> None:
    out = _executor().execute(
        "run_backtest", {"ticker": "TST", "horizon_days": 20, "model": "historical_sim"}
    )
    assert "error" not in out
    assert out["model"] == "historical_sim"
    assert out["n_oos_forecasts"] > 0
    assert 0.0 <= out["mean_brier"] <= 1.0
    assert out["calibration_quality"] in {
        "well-calibrated",
        "moderately calibrated",
        "poorly calibrated",
    }
    assert len(out["per_threshold"]) == 5
    assert {"threshold_label", "brier", "roc_auc"} <= set(out["per_threshold"][0])


def test_get_calibration_tool_shape() -> None:
    out = _executor().execute(
        "get_calibration", {"ticker": "TST", "horizon_days": 20, "model": "monte_carlo_gbm"}
    )
    assert "error" not in out
    assert 0.0 <= out["ece"] <= 1.0
    assert out["calibration_quality"] in {
        "well-calibrated",
        "moderately calibrated",
        "poorly calibrated",
    }
    assert out["reliability"]  # non-empty reliability table
    assert {"predicted", "realized", "count"} <= set(out["reliability"][0])


# ---- bounds + model restriction (cost guards) ---------------------------------
def test_horizon_out_of_bounds_is_rejected() -> None:
    out = _executor().execute("run_backtest", {"ticker": "TST", "horizon_days": 200})
    assert "error" in out and "horizon_days" in out["error"]


def test_ml_model_is_rejected_with_cli_hint() -> None:
    # ML models need an offline per-fold retrain → rejected by the chat backtest tools.
    out = _executor().execute("get_calibration", {"ticker": "TST", "model": "lightgbm"})
    assert "error" in out
    assert "offline" in out["error"].lower() or "cli" in out["error"].lower()


# ---- in-session caching (run both → one computation) --------------------------
def test_backtest_is_cached_across_tools() -> None:
    ex = _executor()
    ex.execute("run_backtest", {"ticker": "TST", "horizon_days": 20, "model": "historical_sim"})
    ex.execute("get_calibration", {"ticker": "TST", "horizon_days": 20, "model": "historical_sim"})
    # Same (ticker, horizon, model) → a single cached BacktestResult, not two.
    assert len(ex._backtest_cache) == 1


# ---- agent loop answers a calibration question with grounded numbers ----------
def test_agent_reports_calibration_grounded() -> None:
    ex = _executor()
    # Compute the real ECE first (also primes the executor's cache for the loop).
    cal = ex.execute("get_calibration", {"ticker": "TST", "horizon_days": 20})
    ece, quality = cal["ece"], cal["calibration_quality"]

    script = [
        ToolResponse(
            text="",
            tool_uses=[
                ToolUse(id="1", name="get_calibration", input={"ticker": "TST", "horizon_days": 20})
            ],
            stop_reason="tool_use",
            assistant_content=[],
        ),
        _final(f"The historical_sim 20-day forecast is {quality}, with an ECE of {ece}."),
    ]
    result = run_agent(
        "Is your 20-day TST forecast well-calibrated?", llm=_Scripted(script), executor=ex
    )
    assert "get_calibration" in result.tool_calls
    assert quality in result.text  # honest trust label surfaced; no grounding error raised


class _Scripted:
    def __init__(self, responses: list[ToolResponse]) -> None:
        self._responses = responses
        self.calls = 0

    def create(self, *, system, messages, tools, max_tokens) -> ToolResponse:  # type: ignore[no-untyped-def]
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return resp


# ---- get_large_move tool (the big-move build) ---------------------------------
def test_get_large_move_tool_shape() -> None:
    # Default model=logistic; falls back to the baseline if no artifact, so the shape
    # assertions hold either way (the breakdown is a pure reading of the bucket distribution).
    out = _executor().execute(
        "get_large_move",
        {"ticker": "TST", "horizon_days": 20, "threshold_pct": 10},
    )
    assert "error" not in out
    assert 0.0 <= out["prob_large_move"] <= 1.0
    # P(|r|>k) must equal P(up) + P(down).
    assert out["prob_large_move"] == pytest.approx(out["prob_big_up"] + out["prob_big_down"])
    assert out["lean"] in {"up", "down", "balanced"}
    assert out["threshold_pct"] == 10
    assert "reliability" in out  # honest trust framing present


def test_get_large_move_rejects_bad_threshold() -> None:
    # At h20 the valid edges are 5 and 10; 7 is not a boundary.
    out = _executor().execute("get_large_move", {"ticker": "TST", "threshold_pct": 7})
    assert "error" in out and "[5, 10]" in out["error"]


def test_get_large_move_default_k_scales_with_horizon() -> None:
    # Omitting threshold_pct uses the horizon's inner boundary: 5 at h20, 15 at h60.
    ex = _executor()
    out20 = ex.execute("get_large_move", {"ticker": "TST", "horizon_days": 20})
    assert out20.get("threshold_pct") == 5
    out60 = ex.execute("get_large_move", {"ticker": "TST", "horizon_days": 60})
    assert out60.get("threshold_pct") == 15


def test_get_large_move_rejects_non_ml_model() -> None:
    # Only the two validated big-move ML models are allowed (logistic / lightgbm).
    out = _executor().execute("get_large_move", {"ticker": "TST", "model": "historical_sim"})
    assert "error" in out and "logistic" in out["error"] and "lightgbm" in out["error"]
