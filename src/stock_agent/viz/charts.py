"""Tool-result -> chart specs for the chat UI (presentation data, no rendering).

The chat agent (Role B) is text-only; it never produces chart data. These pure
builders turn the *numbers the tools already computed* (forecast buckets,
sentiment shares, calibration reliability, ...) into renderable ``ChartSpec``s,
preserving the numbers-vs-narrative invariant: every plotted value traces to a
tool result, never to the LLM.

Pure (pandas only) and decoupled from ``agent`` — ``charts_for`` accepts anything
with ``.name`` and ``.result`` (structural ``_Invocation``), so it does not import
the runtime. Rendering (Altair/Streamlit) lives in the UI layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class ChartSpec:
    """A render-agnostic chart: the data plus hints the UI maps to an Altair mark."""

    title: str
    kind: str  # "bar" | "grouped_bar" | "reliability"
    data: pd.DataFrame
    x: str
    y: str
    caption: str = ""
    color: str | None = None  # grouping column (grouped_bar only)
    x_sort: tuple[str, ...] | None = None  # explicit category order (Altair sorts alpha otherwise)
    y_is_percent: bool = False  # format/scale y as a percentage in the UI


class _Invocation(Protocol):
    """Structural view of a tool call (matches agent.runtime.ToolInvocation)."""

    name: str
    result: dict[str, Any]


def _ok(result: dict[str, Any]) -> bool:
    return "error" not in result


# ---- per-tool builders --------------------------------------------------------
def _forecast_chart(results: list[dict[str, Any]]) -> ChartSpec | None:
    """Scenario bucket probabilities; grouped across models when several were run."""
    rows: list[dict[str, Any]] = []
    order: list[str] = []
    models: list[str] = []
    horizon: Any = None
    for r in results:
        if not _ok(r) or not r.get("buckets"):
            continue
        model = str(r.get("model_name", "model"))
        models.append(model)
        horizon = r.get("horizon_days", horizon)
        for b in r["buckets"]:
            rows.append({"bucket": b["label"], "model": model, "probability": b["probability"]})
            if b["label"] not in order:
                order.append(b["label"])
    if not rows:
        return None
    multi = len(set(models)) > 1
    htxt = f" ({horizon}-day)" if horizon is not None else ""
    return ChartSpec(
        title=f"Forecast scenario probabilities{htxt}",
        kind="grouped_bar" if multi else "bar",
        data=pd.DataFrame(rows),
        x="bucket",
        y="probability",
        color="model" if multi else None,
        x_sort=tuple(order),
        y_is_percent=True,
        caption="Probability the return lands in each range (from the model's buckets).",
    )


def _large_move_chart(r: dict[str, Any]) -> ChartSpec | None:
    """P(|r|>k) split into up-tail / no-big-move / down-tail."""
    if not _ok(r):
        return None
    up, down, total = r.get("prob_big_up"), r.get("prob_big_down"), r.get("prob_large_move")
    if up is None or down is None or total is None:
        return None
    k = r.get("threshold_pct") or int(round(float(r.get("threshold", 0.10)) * 100))
    labels = (f"Big up (>+{k}%)", "No big move", f"Big down (<-{k}%)")
    df = pd.DataFrame({"outcome": list(labels), "probability": [up, max(0.0, 1.0 - total), down]})
    htxt = f", {r['horizon_days']}-day" if r.get("horizon_days") is not None else ""
    return ChartSpec(
        title=f"Large-move breakdown (±{k}%{htxt})",
        kind="bar",
        data=df,
        x="outcome",
        y="probability",
        x_sort=labels,
        y_is_percent=True,
        caption="P(|return| > k) split by tail; 'No big move' is the complement.",
    )


def _sentiment_chart(r: dict[str, Any]) -> ChartSpec | None:
    """Composition of articles by sentiment (positive / neutral / negative shares)."""
    if not _ok(r):
        return None
    pos, neg = r.get("pct_positive"), r.get("pct_negative")
    if pos is None or neg is None:
        return None
    labels = ("Positive", "Neutral", "Negative")
    df = pd.DataFrame({"label": list(labels), "share": [pos, max(0.0, 1.0 - pos - neg), neg]})
    n = int(r.get("article_count", 0) or 0)
    src = r.get("sentiment_source", "")
    return ChartSpec(
        title="News sentiment composition",
        kind="bar",
        data=df,
        x="label",
        y="share",
        x_sort=labels,
        y_is_percent=True,
        caption=f"Share of {n} article(s) by sentiment (source: {src}).",
    )


def _news_insights_chart(r: dict[str, Any]) -> ChartSpec | None:
    """Counts of extracted insight points by category (from a NewsSummary)."""
    keys = ("bullish", "bearish", "risks", "catalysts")
    if not any(isinstance(r.get(k), list) for k in keys):
        return None
    counts = [len(r.get(k) or []) for k in keys]
    if sum(counts) == 0:
        return None
    labels = ("Bullish", "Bearish", "Risks", "Catalysts")
    df = pd.DataFrame({"category": list(labels), "count": counts})
    return ChartSpec(
        title="News insights by category",
        kind="bar",
        data=df,
        x="category",
        y="count",
        x_sort=labels,
        caption="Number of cited points the summary extracted in each category.",
    )


def _calibration_chart(r: dict[str, Any]) -> ChartSpec | None:
    """Reliability diagram: predicted probability vs realized frequency."""
    rel = r.get("reliability")
    if not _ok(r) or not rel:
        return None
    df = pd.DataFrame([{"predicted": b["predicted"], "realized": b["realized"]} for b in rel])
    return ChartSpec(
        title="Calibration reliability (predicted vs realized)",
        kind="reliability",
        data=df,
        x="predicted",
        y="realized",
        caption="Points on the dashed diagonal are perfectly calibrated; below it = overconfident.",
    )


def _backtest_chart(r: dict[str, Any]) -> ChartSpec | None:
    """Out-of-sample ROC AUC per return threshold (0.5 = no skill)."""
    pt = r.get("per_threshold")
    if not _ok(r) or not pt:
        return None
    rows = [
        {"threshold": m["threshold_label"], "roc_auc": m["roc_auc"]}
        for m in pt
        if m.get("roc_auc") is not None
    ]
    if not rows:
        return None
    return ChartSpec(
        title="Out-of-sample ROC AUC by threshold",
        kind="bar",
        data=pd.DataFrame(rows),
        x="threshold",
        y="roc_auc",
        caption="0.5 = no skill; higher = better discrimination at that return threshold.",
    )


# Single-result builders (run_forecast is special-cased to group across models).
_SINGLE: dict[str, Any] = {
    "get_large_move": _large_move_chart,
    "get_news_sentiment": _sentiment_chart,
    "summarize_news": _news_insights_chart,
    "get_calibration": _calibration_chart,
    "run_backtest": _backtest_chart,
}


def charts_for(invocations: Sequence[_Invocation]) -> list[ChartSpec]:
    """Build the charts implied by a turn's tool calls (empty when nothing is plottable).

    ``run_forecast`` calls are pooled into one (possibly grouped) bucket chart so a
    model comparison renders side by side; other tools yield one chart per distinct
    (ticker, horizon, threshold) call.
    """
    specs: list[ChartSpec] = []

    forecasts = [t.result for t in invocations if t.name == "run_forecast"]
    if forecasts:
        spec = _forecast_chart(forecasts)
        if spec is not None:
            specs.append(spec)

    seen: set[tuple[Any, ...]] = set()
    for t in invocations:
        builder = _SINGLE.get(t.name)
        if builder is None:
            continue
        key = (
            t.name,
            t.result.get("ticker"),
            t.result.get("horizon_days"),
            t.result.get("threshold_pct"),
        )
        if key in seen:
            continue
        seen.add(key)
        spec = builder(t.result)
        if spec is not None:
            specs.append(spec)

    return specs
