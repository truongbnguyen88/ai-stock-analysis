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
    direction: str | None = None  # "bar" only: column of "up"/"down"/"neutral" tinting each bar
    # green/red/brass. Set ONLY from tool numbers (e.g. the get_large_move up-/down-tail split),
    # never the LLM — so the §2 signaling rule holds (color = figure, not narrative).

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form (for persisting a chat thread); DataFrame -> column lists."""
        return {
            "title": self.title,
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "caption": self.caption,
            "color": self.color,
            "x_sort": list(self.x_sort) if self.x_sort is not None else None,
            "y_is_percent": self.y_is_percent,
            "direction": self.direction,
            "data": self.data.to_dict(orient="list"),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChartSpec:
        """Inverse of :meth:`to_dict` (rebuilds the DataFrame, preserving columns)."""
        return cls(
            title=d["title"],
            kind=d["kind"],
            data=pd.DataFrame(d["data"]),
            x=d["x"],
            y=d["y"],
            caption=d.get("caption", ""),
            color=d.get("color"),
            x_sort=tuple(d["x_sort"]) if d.get("x_sort") else None,
            y_is_percent=bool(d.get("y_is_percent", False)),
            direction=d.get("direction"),
        )


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


def _research_forecast_charts(r: dict[str, Any]) -> list[ChartSpec]:
    """One scenario-probability bar chart per horizon from the executive brief.

    The brief runs a single model (the ensemble) across several horizons, and each
    horizon has its OWN return bands (buckets scale with horizon — ±5/10% at 20d vs
    ±15/30% at 60d; see forecasting.buckets), so a shared x-axis would misalign. We
    emit a separate chart per horizon using that horizon's real bucket labels — the
    same visual as ``_forecast_chart`` for a single ``run_forecast``. Numbers are the
    model's bucket probabilities, never the LLM's.
    """
    if not _ok(r):
        return []
    specs: list[ChartSpec] = []
    for e in r.get("forecast_buckets") or []:
        buckets = e.get("buckets") or []
        if not buckets:
            continue  # e.g. a horizon the model skipped — nothing to plot
        model = str(e.get("model_name", "model"))
        h = e.get("horizon_days")
        htxt = f", {h}-day" if h is not None else ""
        order = [b["label"] for b in buckets]
        rows = [{"bucket": b["label"], "probability": b["probability"]} for b in buckets]
        specs.append(
            ChartSpec(
                title=f"Forecast scenario probabilities ({model}{htxt})",
                kind="bar",
                data=pd.DataFrame(rows),
                x="bucket",
                y="probability",
                x_sort=tuple(order),
                y_is_percent=True,
                caption="Probability the return lands in each range (from the ensemble's buckets).",
            )
        )
    return specs


def _research_news_charts(r: dict[str, Any]) -> list[ChartSpec]:
    """Recent-news charts for the executive brief: insight counts + sentiment composition.

    Reads the brief's ``news`` block (a NewsAnalysis dump). Each chart no-ops when its
    inputs are absent — the insight chart needs at least one cited point; the sentiment
    chart needs provider scores (``pct_positive``/``pct_negative`` are ``None`` otherwise).
    """
    if not _ok(r):
        return []
    news = r.get("news")
    if not isinstance(news, dict):
        return []
    specs: list[ChartSpec] = []

    keys = ("bullish", "bearish", "risks", "catalysts")
    counts = [len(news.get(k) or []) for k in keys]
    if sum(counts) > 0:
        cat_labels = ("Bullish", "Bearish", "Risks", "Catalysts")
        days = news.get("lookback_days") or 0
        specs.append(
            ChartSpec(
                title="Recent-news insights by category",
                kind="bar",
                data=pd.DataFrame({"category": list(cat_labels), "count": counts}),
                x="category",
                y="count",
                x_sort=cat_labels,
                caption=f"Number of cited recent-news points in each category (last {days}d).",
            )
        )

    pos, neg = news.get("pct_positive"), news.get("pct_negative")
    if pos is not None and neg is not None:
        sent_labels = ("Positive", "Neutral", "Negative")
        # Neutral = complement; it also absorbs unscored articles (see sentiment_coverage).
        df = pd.DataFrame(
            {"label": list(sent_labels), "share": [pos, max(0.0, 1.0 - pos - neg), neg]}
        )
        n = int(news.get("article_count", 0) or 0)
        cov = news.get("sentiment_coverage")
        cov_txt = f"; {cov:.0%} of articles scored" if cov is not None else ""
        specs.append(
            ChartSpec(
                title="Recent-news sentiment composition",
                kind="bar",
                data=df,
                x="label",
                y="share",
                x_sort=sent_labels,
                y_is_percent=True,
                caption=f"Share of {n} article(s) by sentiment{cov_txt}.",
            )
        )
    return specs


def _research_large_move_chart(r: dict[str, Any]) -> ChartSpec | None:
    """Large-move tail split for the executive brief (reads the consolidated ``large_move`` block).

    The brief's ``large_move`` sub-dict is a ``LargeMoveBreakdown`` dump — the same shape
    ``_large_move_chart`` reads for a standalone ``get_large_move`` — so reuse that builder.
    """
    if not _ok(r):
        return None
    lm = r.get("large_move")
    if not isinstance(lm, dict):
        return None
    return _large_move_chart(lm)


def _large_move_chart(r: dict[str, Any]) -> ChartSpec | None:
    """P(|r|>k) split into up-tail / no-big-move / down-tail."""
    if not _ok(r):
        return None
    up, down, total = r.get("prob_big_up"), r.get("prob_big_down"), r.get("prob_large_move")
    if up is None or down is None or total is None:
        return None
    k = r.get("threshold_pct") or int(round(float(r.get("threshold", 0.10)) * 100))
    labels = (f"Big up (>+{k}%)", "No big move", f"Big down (<-{k}%)")
    # Per-bar direction from the tool's own tail split (up-tail green, down-tail red, middle
    # neutral) — the numbers came from the model, so tinting them honors the §2 signaling rule.
    directions = ("up", "neutral", "down")
    df = pd.DataFrame(
        {
            "outcome": list(labels),
            "probability": [up, max(0.0, 1.0 - total), down],
            "direction": list(directions),
        }
    )
    htxt = f", {r['horizon_days']}-day" if r.get("horizon_days") is not None else ""
    return ChartSpec(
        title=f"Large-move breakdown (±{k}%{htxt})",
        kind="bar",
        data=df,
        x="outcome",
        y="probability",
        x_sort=labels,
        y_is_percent=True,
        direction="direction",
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


# ---- multi-ticker comparisons (Enhancement B) ---------------------------------
def _compare_forecasts_chart(r: dict[str, Any]) -> ChartSpec | None:
    """Grouped bars across tickers: P(up) and P(big move) per name (errored rows skipped)."""
    if not _ok(r):
        return None
    rows: list[dict[str, Any]] = []
    for row in r.get("rows") or []:
        if row.get("error"):
            continue
        ticker = row.get("ticker")
        up, big = row.get("upside_prob"), row.get("prob_large_move")
        if up is not None:
            rows.append({"ticker": ticker, "metric": "P(up)", "probability": up})
        if big is not None:
            rows.append({"ticker": ticker, "metric": "P(big move)", "probability": big})
    if not rows:
        return None
    horizon = r.get("horizon_days")
    htxt = f", {horizon}-day" if horizon is not None else ""
    return ChartSpec(
        title=f"Forecast comparison ({r.get('model', 'model')}{htxt})",
        kind="grouped_bar",
        data=pd.DataFrame(rows),
        x="ticker",
        y="probability",
        color="metric",
        x_sort=tuple(dict.fromkeys(row["ticker"] for row in rows)),
        y_is_percent=True,
        caption="P(up) and P(|return|>k) per ticker — from the model, not the assistant.",
    )


def _compare_news_chart(r: dict[str, Any]) -> ChartSpec | None:
    """Grouped bars across tickers: positive vs negative article share per name."""
    if not _ok(r):
        return None
    rows: list[dict[str, Any]] = []
    for row in r.get("rows") or []:
        if row.get("error"):
            continue
        ticker = row.get("ticker")
        pos, neg = row.get("pct_positive"), row.get("pct_negative")
        if pos is None or neg is None:
            continue
        rows.append({"ticker": ticker, "sentiment": "Positive", "share": pos})
        rows.append({"ticker": ticker, "sentiment": "Negative", "share": neg})
    if not rows:
        return None
    days = r.get("days")
    dtxt = f", last {days}d" if days is not None else ""
    return ChartSpec(
        title=f"News sentiment comparison{dtxt}",
        kind="grouped_bar",
        data=pd.DataFrame(rows),
        x="ticker",
        y="share",
        color="sentiment",
        x_sort=tuple(dict.fromkeys(row["ticker"] for row in rows)),
        y_is_percent=True,
        caption="Positive vs negative article share per ticker (from sentiment scores).",
    )


def _conditional_chart(r: dict[str, Any]) -> ChartSpec | None:
    """Mean forward return after a driver shock vs the unconditional baseline (#7)."""
    if not _ok(r):
        return None
    base = r.get("baseline")
    cond = r.get("conditional")
    if not base:
        return None
    rows: list[dict[str, Any]] = []
    if cond:
        rows.append({"scenario": "After driver shock", "mean_return": cond["mean"]})
    rows.append({"scenario": "Baseline (all days)", "mean_return": base["mean"]})
    target, driver = r.get("target", "?"), r.get("driver", "?")
    htxt = f", {r['horizon_days']}-day" if r.get("horizon_days") is not None else ""
    return ChartSpec(
        title=f"{target} forward return after {driver} shocks vs baseline{htxt}",
        kind="bar",
        data=pd.DataFrame(rows),
        x="scenario",
        y="mean_return",
        x_sort=tuple(row["scenario"] for row in rows),
        y_is_percent=True,
        caption="Conditional vs baseline mean forward return (descriptive, not a forecast).",
    )


def _topic_news_chart(r: dict[str, Any]) -> ChartSpec | None:
    """Theme news insight counts, mirroring the per-ticker summary chart.

    Topic synthesis (analyze_topic_news) carries the same bullish/bearish/risks/
    catalysts structure as a ticker summary, so reuse that builder. Headlines and
    the (usually source-absent) topic tone line render as text, not a chart.
    """
    if not _ok(r):
        return None
    return _news_insights_chart(r)


# Single-result builders (run_forecast is special-cased to group across models).
_SINGLE: dict[str, Any] = {
    "get_large_move": _large_move_chart,
    "get_news_sentiment": _sentiment_chart,
    "summarize_news": _news_insights_chart,
    "get_calibration": _calibration_chart,
    "run_backtest": _backtest_chart,
    "compare_forecasts": _compare_forecasts_chart,
    "compare_news": _compare_news_chart,
    "analyze_topic_news": _topic_news_chart,
    "conditional_outlook": _conditional_chart,
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

    # The executive brief (research_summary) carries per-horizon buckets + a news block +
    # a large-move split → one forecast chart per horizon, the news insight/sentiment charts,
    # then the large-move tail chart.
    for t in invocations:
        if t.name == "research_summary":
            specs.extend(_research_forecast_charts(t.result))
            specs.extend(_research_news_charts(t.result))
            lm_spec = _research_large_move_chart(t.result)
            if lm_spec is not None:
                specs.append(lm_spec)

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
