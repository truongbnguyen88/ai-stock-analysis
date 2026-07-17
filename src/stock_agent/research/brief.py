"""Deterministic rich-markdown renderer for the integrated research brief.

Turns a :class:`ResearchMemo` into the executive-brief Markdown the chat UI presents for
``research_summary`` — numbered ``##`` sections (brass headers in the web app), pipe tables for the
quantitative blocks (price / technicals / ensemble forecasts / large-move / earnings), and bulleted
qualitative sections (news themes, filing drivers/risks, bull vs bear evidence, uncertainties).

Why deterministic (not LLM-composed): every figure is laid out in code straight from the memo's
numeric fields, so the numbers-vs-narrative invariant holds *by construction* — the LLM never
transcribes a figure into a table cell (which is exactly where a re-round/slip would trip the
number-grounding guard and blank the turn). The narrative prose is the memo's own grounded synthesis
(``executive_summary`` and the driver/risk/evidence/uncertainty lists), already citation- and
number-guarded upstream. So this function does layout only; it computes no new quantity and states
no recommendation (DESIGN INVARIANT §2).

Signal labels in the technicals table (e.g. "price above — bullish", "oversold") are deterministic
descriptors of the *current* indicator state, not forecasts.
"""

from __future__ import annotations

from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.research import NewsAnalysis, ResearchMemo

# Ordered (key, label, kind) for the technicals table. `kind` selects value formatting + the
# deterministic signal descriptor. `last_close` / `last_daily_return` are intentionally omitted —
# they live in the Price Snapshot table; `last_close` is only used here to sign the MA rows.
_TECH_ROWS: tuple[tuple[str, str, str], ...] = (
    ("ma20", "MA20", "ma"),
    ("ma50", "MA50", "ma"),
    ("ma200", "MA200", "ma"),
    ("rsi14", "RSI (14)", "rsi"),
    ("macd", "MACD", "macd"),
    ("macd_signal", "MACD signal", "plain"),
    ("macd_hist", "MACD histogram", "hist"),
    ("hist_vol_annualized", "Annualized volatility", "vol"),
    ("atr14", "ATR (14)", "usd"),
    ("max_drawdown", "Max drawdown", "dd"),
)


def _tech_signal(kind: str, value: float, ind: dict[str, float]) -> str:
    """A deterministic, non-predictive descriptor of one indicator's current state."""
    if kind == "ma":
        last_close = ind.get("last_close")
        if last_close is None:
            return ""
        return "price above — bullish" if last_close >= value else "price below — bearish"
    if kind == "macd":
        sig = ind.get("macd_signal")
        if sig is None:
            return ""
        return "above signal — bullish" if value >= sig else "below signal — bearish"
    if kind == "usd":  # ATR
        return "average daily range"
    if kind == "rsi":
        if value <= 30:
            return "oversold"
        if value >= 70:
            return "overbought"
        if value < 45:
            return "soft (below midpoint)"
        if value > 55:
            return "firm (above midpoint)"
        return "neutral (near midpoint)"
    if kind == "hist":
        return "positive — momentum building" if value >= 0 else "negative — momentum fading"
    if kind == "vol":  # annualized fraction
        if value >= 1.0:
            return "extremely elevated"
        if value >= 0.5:
            return "elevated"
        return "moderate"
    if kind == "dd":
        return "peak-to-trough risk"
    return ""


def _tech_value(kind: str, value: float) -> str:
    """Format one indicator value by kind (prices as USD, RSI/MACD raw, vol/drawdown as %)."""
    if kind in ("ma", "usd"):
        return f"${value:,.2f}"
    if kind == "vol":
        return f"{value:.1%}"
    if kind == "dd":
        return f"{value:+.1%}"
    if kind == "hist":
        return f"{value:+.2f}"
    return f"{value:.2f}"  # rsi, macd, macd_signal, plain


def _trend_label(ind: dict[str, float]) -> str | None:
    """Long-term regime descriptor from price vs MA50/MA200 (mirrors indicators.snapshot)."""
    last_close, ma50, ma200 = ind.get("last_close"), ind.get("ma50"), ind.get("ma200")
    if last_close is None or ma50 is None:
        return None
    if last_close > ma50 and (ma200 is None or ma50 > ma200):
        return "Uptrend"
    if last_close < ma50 and (ma200 is None or ma50 < ma200):
        return "Downtrend"
    return "Sideways"


def _bullets(items: list[str]) -> list[str]:
    """Render non-empty strings as Markdown bullets (already-embedded [n] markers preserved)."""
    return [f"- {s.strip()}" for s in items if s and s.strip()]


def _price_section(memo: ResearchMemo) -> list[str]:
    ps = memo.price_snapshot
    if ps is None:
        return []
    rows = [
        ("Last close", f"${ps.last_close:,.2f}"),
        (f"{ps.window_days}-day high", f"${ps.period_high:,.2f}"),
        (f"{ps.window_days}-day low", f"${ps.period_low:,.2f}"),
        ("Period return", f"{ps.pct_change:+.2%}"),
    ]
    if ps.last_return is not None:
        rows.append(("Last daily return", f"{ps.last_return:+.2%}"))
    out = [
        f"## 📊 1. Price Snapshot (last {ps.window_days}d, {ps.n_bars} bars)",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    out += [f"| {k} | {v} |" for k, v in rows]
    return out


def _tech_section(memo: ResearchMemo) -> list[str]:
    ind = memo.technical_indicators
    if not ind:
        return []
    rows: list[str] = []
    for key, label, kind in _TECH_ROWS:
        if key not in ind:
            continue
        val, sig = _tech_value(kind, ind[key]), _tech_signal(kind, ind[key], ind)
        rows.append(f"| {label} | {val} | {sig} |")
    trend = _trend_label(ind)
    if trend is not None:
        rows.append(f"| Trend | **{trend}** | long-term structure |")
    if not rows:  # no recognized indicators → omit rather than emit a header-only table
        return []
    head = ["## 📈 2. Technical Indicators", "", "| Indicator | Value | Signal |", "|---|---|---|"]
    return [*head, *rows]


def _forecast_row(fc: ScenarioForecast) -> str:
    """One horizon's row: P(up) · E[r] · VaR95 · CI (missing pieces rendered as em dash)."""
    var = f"{fc.var_95:.1%}" if fc.var_95 is not None else "—"
    if fc.ci_low is not None and fc.ci_high is not None:
        ci = f"[{fc.ci_low:+.1%}, {fc.ci_high:+.1%}]"
    else:
        ci = "—"
    return (
        f"| {fc.horizon_days} days | {fc.upside_prob:.1%} | {fc.expected_return:+.1%} "
        f"| {var} | {ci} |"
    )


def _forecast_section(memo: ResearchMemo) -> list[str]:
    if not memo.forecasts:
        return []
    forecasts = sorted(memo.forecasts, key=lambda f: f.horizon_days)
    model = forecasts[0].model_name
    ci_level = next((f.ci_level for f in forecasts if f.ci_level is not None), None)
    ci_hdr = f"{ci_level:.0%} CI" if ci_level is not None else "CI"
    out = [
        "## 🔮 3. Model Forecasts (Ensemble)",
        "",
        f"> *Figures are outputs of the `{model}` model — a calibrated probability pool of "
        "historical-simulation, Monte-Carlo, GARCH, logistic, and gradient-boosted members. "
        "Model estimates, not guarantees; treat wide intervals as a scenario distribution.*",
        "",
        f"| Horizon | P(up) | Expected return | VaR 95% | {ci_hdr} |",
        "|---|---|---|---|---|",
    ]
    out += [_forecast_row(fc) for fc in forecasts]
    return out


def _large_move_section(memo: ResearchMemo) -> list[str]:
    lm = memo.large_move
    if lm is None:
        return []
    thr = f"{lm.threshold:.0%}"
    return [
        f"## 💥 4. Large-Move Probability ({lm.horizon_days}d, ±{thr})",
        "",
        "| Metric | Value |",
        "|---|---|",
        # Escape the pipes in |return| so GFM does not read them as column separators.
        f"| P(\\|return\\| > {thr}) | **{lm.prob_large_move:.0%}** |",
        f"| P(up > +{thr}) | {lm.prob_big_up:.0%} |",
        f"| P(down < -{thr}) | {lm.prob_big_down:.0%} |",
        f"| Lean | **{lm.lean}** |",
    ]


def _earnings_section(memo: ResearchMemo) -> list[str]:
    ea = memo.earnings
    if ea is None or not (ea.next_earnings_date or ea.last_earnings_date):
        return []
    out = ["## 📅 5. Earnings Context", "", "| Item | Detail |", "|---|---|"]
    if ea.last_earnings_date:
        since = f" ({ea.days_since_last_earnings} days ago)" if ea.days_since_last_earnings else ""
        out.append(f"| Last earnings | {ea.last_earnings_date.isoformat()}{since} |")
    if ea.next_earnings_date:
        away = f" ({ea.days_to_next_earnings} days away)" if ea.days_to_next_earnings else ""
        out.append(f"| Next earnings | {ea.next_earnings_date.isoformat()}{away} |")
    if ea.earnings_in_horizon is not None and ea.horizon_days:
        flag = "✅ Yes — unmodeled event risk" if ea.earnings_in_horizon else "❌ No"
        out.append(f"| In {ea.horizon_days}-day window? | {flag} |")
    return out


def _news_section(memo: ResearchMemo) -> list[str]:
    na: NewsAnalysis | None = memo.news
    if na is None or na.article_count == 0:
        return []
    out = [f"## 📰 6. News & Sentiment (last {na.lookback_days}d)", ""]
    if na.avg_sentiment is not None and na.pct_positive is not None and na.pct_negative is not None:
        out.append(
            f"**Sentiment:** avg {na.avg_sentiment:+.3f} · {na.pct_positive:.0%} positive / "
            f"{na.pct_negative:.0%} negative · {na.article_count} articles"
        )
    elif na.overview:
        out.append(na.overview)
    for hdr, pts in (("### 🟢 Bullish themes", na.bullish), ("### 🔴 Bearish themes", na.bearish),
                     ("### 📌 Catalysts", na.catalysts), ("### ⚠️ Risks", na.risks)):
        bullets = _bullets(pts)
        if bullets:
            out += ["", hdr, *bullets]
    return out


def _drivers_risks_section(memo: ResearchMemo) -> list[str]:
    drivers, risks = _bullets(memo.business_drivers), _bullets(memo.risk_factors)
    if not (drivers or risks or memo.management_commentary):
        return []
    out = ["## 🏢 7. Business Drivers & Risk Factors (SEC filings)"]
    if drivers:
        out += ["", "### Business drivers", *drivers]
    if risks:
        out += ["", "### Risk factors", *risks]
    if memo.management_commentary:
        out += ["", "### Management commentary", memo.management_commentary.strip()]
    return out


def _bull_bear_section(memo: ResearchMemo) -> list[str]:
    bull, bear = _bullets(memo.bullish_evidence), _bullets(memo.bearish_evidence)
    if not (bull or bear):
        return []
    out = ["## ⚖️ 8. Bull vs Bear"]
    if bull:
        out += ["", "### 🟢 Bullish evidence", *bull]
    if bear:
        out += ["", "### 🔴 Bearish evidence", *bear]
    return out


def _uncertainty_section(memo: ResearchMemo) -> list[str]:
    notes = _bullets(memo.uncertainty_notes)
    if not notes:
        return []
    return ["## 🔑 9. Key Uncertainties", "", *notes]


def _sources_section(memo: ResearchMemo) -> list[str]:
    if not memo.citations:
        return []
    return ["## 🗂 Source Citations", "", *[f"- [{c.marker}] {c.label}" for c in memo.citations]]


_DISCLAIMER = (
    "*Research and education only. Not financial advice, not a recommendation, no price target. "
    "All quantitative figures are statistical model outputs or filing excerpts carrying "
    "significant uncertainty; past model performance does not guarantee future accuracy.*"
)


def render_research_brief(memo: ResearchMemo) -> str:
    """Render a ``ResearchMemo`` as the rich executive-brief Markdown (tables + brass sections).

    Sections are emitted only when their data exists, so a ticker with no filings/news/earnings
    still produces a clean brief. Numbers come verbatim from the memo's numeric fields (invariant);
    narrative comes from its grounded synthesis. Returns a single Markdown string.
    """
    parts: list[list[str]] = [
        [
            f"# 🔬 {memo.ticker} — Executive Research Brief",
            f"*As of {memo.as_of:%b %d, %Y} · Research & education only — not financial advice*",
        ],
        [
            "## 📌 Executive Summary",
            "",
            memo.executive_summary.strip() or "_No summary available._",
        ],
        _price_section(memo),
        _tech_section(memo),
        _forecast_section(memo),
        _large_move_section(memo),
        _earnings_section(memo),
        _news_section(memo),
        _drivers_risks_section(memo),
        _bull_bear_section(memo),
        _uncertainty_section(memo),
        _sources_section(memo),
        [_DISCLAIMER],
    ]
    # Join sections with a blank line between them; drop sections that rendered empty.
    return "\n\n".join("\n".join(block) for block in parts if block)
