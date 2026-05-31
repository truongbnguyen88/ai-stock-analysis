"""Pure chart-builder tests: tool-result dicts -> ChartSpec (no rendering)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stock_agent.viz.charts import ChartSpec, charts_for


@dataclass
class _Inv:
    """Minimal stand-in for agent.runtime.ToolInvocation (structural match)."""

    name: str
    result: dict[str, Any]


def _forecast(model: str, probs: list[float], horizon: int = 20) -> dict[str, Any]:
    labels = ["< -10%", "-10% to -5%", "-5% to 0%", "0% to +5%", "+5% to +10%", "> +10%"]
    return {
        "ticker": "NVDA",
        "horizon_days": horizon,
        "model_name": model,
        "buckets": [{"label": lbl, "probability": p} for lbl, p in zip(labels, probs, strict=True)],
    }


_EVEN = [0.1, 0.1, 0.3, 0.3, 0.1, 0.1]


def _spec_titled(specs: list[ChartSpec], needle: str) -> ChartSpec:
    match = [s for s in specs if needle.lower() in s.title.lower()]
    assert match, f"no chart titled like {needle!r} in {[s.title for s in specs]}"
    return match[0]


# ---- forecast buckets ---------------------------------------------------------
def test_single_forecast_is_a_bar_chart() -> None:
    specs = charts_for([_Inv("run_forecast", _forecast("historical_sim", _EVEN))])
    spec = _spec_titled(specs, "scenario probabilities")
    assert spec.kind == "bar"
    assert spec.color is None
    assert len(spec.data) == 6  # six buckets
    assert spec.x_sort is not None and spec.x_sort[0] == "< -10%"  # explicit order preserved
    assert spec.y_is_percent


def test_multi_model_forecast_is_grouped() -> None:
    specs = charts_for(
        [
            _Inv("run_forecast", _forecast("logistic", _EVEN)),
            _Inv("run_forecast", _forecast("ml_lightgbm", _EVEN)),
        ]
    )
    spec = _spec_titled(specs, "scenario probabilities")
    assert spec.kind == "grouped_bar"
    assert spec.color == "model"
    assert set(spec.data["model"]) == {"logistic", "ml_lightgbm"}
    assert len(spec.data) == 12  # 6 buckets x 2 models, one pooled chart


def test_error_forecast_yields_no_chart() -> None:
    assert charts_for([_Inv("run_forecast", {"error": "nope"})]) == []


# ---- large move ---------------------------------------------------------------
def test_large_move_split_sums_to_one() -> None:
    r = {
        "ticker": "NVDA",
        "horizon_days": 20,
        "prob_big_up": 0.2,
        "prob_big_down": 0.15,
        "prob_large_move": 0.35,
        "threshold_pct": 10,
    }
    spec = _spec_titled(charts_for([_Inv("get_large_move", r)]), "large-move")
    assert spec.kind == "bar"
    assert abs(float(spec.data["probability"].sum()) - 1.0) < 1e-9  # up + none + down = 1
    assert list(spec.data["outcome"])[1] == "No big move"


# ---- sentiment ----------------------------------------------------------------
def test_sentiment_neutral_is_complement() -> None:
    r = {"ticker": "NVDA", "pct_positive": 0.5, "pct_negative": 0.2, "article_count": 10,
         "sentiment_source": "alpha_vantage"}
    spec = _spec_titled(charts_for([_Inv("get_news_sentiment", r)]), "sentiment")
    shares = dict(zip(spec.data["label"], spec.data["share"], strict=True))
    assert abs(shares["Neutral"] - 0.3) < 1e-9
    assert abs(sum(shares.values()) - 1.0) < 1e-9


# ---- news insight counts ------------------------------------------------------
def test_news_insight_counts() -> None:
    r = {
        "ticker": "NVDA",
        "bullish": [{"point": "a"}, {"point": "b"}],
        "bearish": [{"point": "c"}],
        "risks": [],
        "catalysts": [{"point": "d"}],
    }
    spec = _spec_titled(charts_for([_Inv("summarize_news", r)]), "insights")
    counts = dict(zip(spec.data["category"], spec.data["count"], strict=True))
    assert counts == {"Bullish": 2, "Bearish": 1, "Risks": 0, "Catalysts": 1}


def test_empty_news_summary_yields_no_chart() -> None:
    r: dict[str, Any] = {"bullish": [], "bearish": [], "risks": [], "catalysts": []}
    assert charts_for([_Inv("summarize_news", r)]) == []


# ---- calibration reliability --------------------------------------------------
def test_calibration_reliability_chart() -> None:
    r = {
        "ticker": "NVDA",
        "reliability": [
            {"predicted": 0.1, "realized": 0.08, "count": 30},
            {"predicted": 0.5, "realized": 0.55, "count": 20},
        ],
    }
    spec = _spec_titled(charts_for([_Inv("get_calibration", r)]), "reliability")
    assert spec.kind == "reliability"
    assert list(spec.data.columns) == ["predicted", "realized"]


# ---- dedup + ordering ---------------------------------------------------------
def test_duplicate_calls_are_deduped() -> None:
    r: dict[str, Any] = {
        "ticker": "NVDA",
        "horizon_days": 20,
        "pct_positive": 0.4,
        "pct_negative": 0.3,
        "article_count": 5,
        "sentiment_source": "alpha_vantage",
    }
    specs = charts_for([_Inv("get_news_sentiment", r), _Inv("get_news_sentiment", dict(r))])
    assert len([s for s in specs if "sentiment" in s.title.lower()]) == 1
