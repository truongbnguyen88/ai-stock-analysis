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


# ---- executive brief (research_summary): one bucket chart per horizon ----------
def _brief(horizons: list[int]) -> dict[str, Any]:
    """A research_summary tool result carrying per-horizon bucket distributions."""
    labels = ["< -10%", "-10% to -5%", "-5% to 0%", "0% to +5%", "+5% to +10%", "> +10%"]
    return {
        "ticker": "NVDA",
        "forecast_buckets": [
            {
                "model_name": "ensemble",
                "horizon_days": h,
                "buckets": [
                    {"label": lbl, "probability": p}
                    for lbl, p in zip(labels, _EVEN, strict=True)
                ],
            }
            for h in horizons
        ],
    }


def test_brief_emits_one_forecast_chart_per_horizon() -> None:
    specs = charts_for([_Inv("research_summary", _brief([20, 30, 60]))])
    fc_specs = [s for s in specs if "scenario probabilities" in s.title.lower()]
    assert len(fc_specs) == 3  # one per horizon, each with its own scaled bands
    assert all(s.kind == "bar" and s.color is None and s.y_is_percent for s in fc_specs)
    assert all(len(s.data) == 6 for s in fc_specs)  # six buckets each
    # Each chart names its horizon so the three are distinguishable.
    titles = " ".join(s.title for s in fc_specs)
    assert all(f"{h}-day" in titles for h in (20, 30, 60))


def test_brief_skips_horizons_with_no_buckets() -> None:
    r = _brief([20, 60])
    r["forecast_buckets"][1]["buckets"] = []  # e.g. the model skipped this horizon
    specs = charts_for([_Inv("research_summary", r)])
    assert len([s for s in specs if "scenario probabilities" in s.title.lower()]) == 1


def test_brief_with_no_forecast_buckets_yields_no_chart() -> None:
    # Mirrors the _memo() test fixture whose forecast has buckets=[].
    assert charts_for([_Inv("research_summary", {"ticker": "NVDA", "forecast_buckets": []})]) == []
    assert charts_for([_Inv("research_summary", {"error": "boom"})]) == []


# ---- executive brief: recent-news insight + sentiment charts -------------------
def _brief_news() -> dict[str, Any]:
    r = _brief([20])
    r["news"] = {
        "lookback_days": 21,
        "article_count": 10,
        "bullish": ["a", "b"],
        "bearish": ["c"],
        "risks": ["d"],
        "catalysts": [],
        "pct_positive": 0.5,
        "pct_negative": 0.2,
        "sentiment_coverage": 0.4,
    }
    return r


def test_brief_emits_news_insight_and_sentiment_charts() -> None:
    specs = charts_for([_Inv("research_summary", _brief_news())])
    ins = _spec_titled(specs, "insights by category")
    assert ins.kind == "bar"
    counts = dict(zip(ins.data["category"], ins.data["count"], strict=True))
    assert counts == {"Bullish": 2, "Bearish": 1, "Risks": 1, "Catalysts": 0}
    sent = _spec_titled(specs, "sentiment composition")
    assert sent.y_is_percent
    assert abs(float(sent.data["share"].sum()) - 1.0) < 1e-9  # pos + neutral + neg = 1


def test_brief_news_charts_noop_without_scores_or_points() -> None:
    r = _brief([20])
    r["news"] = {
        "lookback_days": 21,
        "article_count": 0,
        "bullish": [],
        "bearish": [],
        "risks": [],
        "catalysts": [],
        "pct_positive": None,  # no provider scores → no sentiment chart
        "pct_negative": None,
        "sentiment_coverage": 0.0,
    }
    specs = charts_for([_Inv("research_summary", r)])
    news_titled = [s for s in specs if "news" in s.title.lower() or "sentiment" in s.title.lower()]
    assert news_titled == []  # forecast chart still present, but no news charts


# ---- executive brief: consolidated large-move chart ----------------------------
def test_brief_emits_large_move_chart_from_consolidated_block() -> None:
    r = _brief([20])
    r["large_move"] = {
        "prob_big_up": 0.47,
        "prob_big_down": 0.46,
        "prob_large_move": 0.93,
        "threshold": 0.05,  # fractional (LargeMoveBreakdown dump) → chart derives ±5%
        "horizon_days": 20,
    }
    spec = _spec_titled(charts_for([_Inv("research_summary", r)]), "large-move")
    assert spec.kind == "bar" and spec.y_is_percent
    assert "±5%" in spec.title
    # up-tail + no-big-move + down-tail sums to 1.
    assert abs(float(spec.data["probability"].sum()) - 1.0) < 1e-9


def test_brief_without_large_move_block_emits_no_large_move_chart() -> None:
    specs = charts_for([_Inv("research_summary", _brief([20]))])  # no "large_move" key
    assert [s for s in specs if "large-move" in s.title.lower()] == []


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
    # Tool-driven per-bar direction (up-tail green / middle neutral / down-tail red).
    assert spec.direction == "direction"
    assert list(spec.data["direction"]) == ["up", "neutral", "down"]


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


# ---- multi-ticker comparisons (Enhancement B) ---------------------------------
def test_compare_forecasts_grouped_by_ticker() -> None:
    r = {
        "horizon_days": 20,
        "model": "ensemble",
        "rows": [
            {"ticker": "NVDA", "upside_prob": 0.6, "prob_large_move": 0.3},
            {"ticker": "MSFT", "upside_prob": 0.55, "prob_large_move": 0.2},
            {"ticker": "BADX", "error": "no data"},  # skipped, no chart rows
        ],
    }
    spec = _spec_titled(charts_for([_Inv("compare_forecasts", r)]), "forecast comparison")
    assert spec.kind == "grouped_bar"
    assert spec.color == "metric"
    assert set(spec.data["ticker"]) == {"NVDA", "MSFT"}  # errored row excluded
    assert set(spec.data["metric"]) == {"P(up)", "P(big move)"}
    assert spec.y_is_percent


def test_compare_news_grouped_by_ticker() -> None:
    r = {
        "days": 14,
        "rows": [
            {"ticker": "NVDA", "pct_positive": 0.5, "pct_negative": 0.2},
            {"ticker": "AMD", "pct_positive": 0.4, "pct_negative": 0.3},
        ],
    }
    spec = _spec_titled(charts_for([_Inv("compare_news", r)]), "sentiment comparison")
    assert spec.kind == "grouped_bar"
    assert set(spec.data["sentiment"]) == {"Positive", "Negative"}
    assert set(spec.data["ticker"]) == {"NVDA", "AMD"}


def test_compare_all_errored_yields_no_chart() -> None:
    r = {"horizon_days": 20, "model": "ensemble", "rows": [{"ticker": "X", "error": "no data"}]}
    assert charts_for([_Inv("compare_forecasts", r)]) == []


def test_topic_news_insight_counts() -> None:
    r = {
        "topic": "robotics",
        "bullish": [{"point": "a"}, {"point": "b"}],
        "bearish": [{"point": "c"}],
        "risks": [],
        "catalysts": [],
        "topic_sentiment": None,
    }
    spec = _spec_titled(charts_for([_Inv("analyze_topic_news", r)]), "insights")
    counts = dict(zip(spec.data["category"], spec.data["count"], strict=True))
    assert counts == {"Bullish": 2, "Bearish": 1, "Risks": 0, "Catalysts": 0}


def test_topic_news_no_insights_yields_no_chart() -> None:
    r = {"topic": "robotics", "error": "no news found for this theme in the window"}
    assert charts_for([_Inv("analyze_topic_news", r)]) == []


def test_conditional_outlook_chart_compares_to_baseline() -> None:
    r = {
        "target": "DAL",
        "driver": "USO",
        "horizon_days": 10,
        "conditional": {"mean": -0.03, "prob_up": 0.4},
        "baseline": {"mean": 0.005, "prob_up": 0.55},
    }
    spec = _spec_titled(charts_for([_Inv("conditional_outlook", r)]), "vs baseline")
    assert spec.kind == "bar"
    assert list(spec.data["scenario"]) == ["After driver shock", "Baseline (all days)"]
    assert spec.y_is_percent


def test_conditional_outlook_no_events_charts_baseline_only() -> None:
    r = {"target": "DAL", "driver": "USO", "conditional": None, "baseline": {"mean": 0.01}}
    spec = _spec_titled(charts_for([_Inv("conditional_outlook", r)]), "vs baseline")
    assert list(spec.data["scenario"]) == ["Baseline (all days)"]


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
