"""Render a ``ResearchReport`` to Markdown.

Rendered in code (no templating dependency for the MVP). Produces every required
section plus a prominent non-advisory disclaimer and a probabilistic-scenarios
table. Numeric fields are formatted for readability; nothing is invented here.
"""

from __future__ import annotations

from stock_agent.forecasting.large_move import large_move_breakdown
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.report import ResearchReport

_DISCLAIMER = (
    "> **Disclaimer:** For research and educational purposes only. "
    "**Not financial advice.** No buy/sell/hold recommendations."
)

# Indicator key -> (display label, formatter).
_PCT_KEYS = {"last_daily_return", "hist_vol_annualized", "max_drawdown"}
_LABELS = {
    "last_close": "Last close",
    "last_daily_return": "Last daily return",
    "ma20": "MA20",
    "ma50": "MA50",
    "ma200": "MA200",
    "rsi14": "RSI(14)",
    "macd": "MACD",
    "macd_signal": "MACD signal",
    "macd_hist": "MACD histogram",
    "hist_vol_annualized": "Annualized volatility",
    "atr14": "ATR(14)",
    "max_drawdown": "Max drawdown",
}


def _fmt_indicator(key: str, value: float) -> str:
    if key in _PCT_KEYS:
        return f"{value:.2%}"
    return f"{value:,.2f}"


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- _None._"]


def _horizon_trust(horizon_days: int) -> str | None:
    """Honest, horizon-dependent confidence caveat (from OOS backtest findings).

    Long horizons have few *independent* (non-overlapping) windows, so OOS skill is
    unmeasurable there regardless of model; short/medium horizons are where the
    big-move probabilities backtest with real skill. Qualitative, no invented numbers.
    """
    if horizon_days >= 60:
        return (
            "long horizon — few independent windows make out-of-sample validation weak; "
            "treat these probabilities as low-confidence"
        )
    if horizon_days <= 30:
        return (
            "horizon with measurable out-of-sample skill in backtests; the big-move "
            "(magnitude) probabilities are the most reliable read"
        )
    return None


def _big_move_line(fc: ScenarioForecast) -> str | None:
    """Horizon-scaled big-move reading at the forecast's inner threshold (ML's niche).

    Derived from the bucket distribution via ``large_move_breakdown`` — no new numbers
    are invented here. ``None`` when the buckets lack a positive boundary to read.
    """
    inner_k = sorted({b.lower for b in fc.buckets if b.lower is not None and b.lower > 0.0})
    if not inner_k:
        return None
    try:
        lm = large_move_breakdown(fc, threshold=inner_k[0])
    except ValueError:
        return None
    return (
        f"- Big move (|r| > {lm.threshold:.0%}): **{lm.prob_large_move:.0%}** "
        f"(up {lm.prob_big_up:.0%} / down {lm.prob_big_down:.0%}, leans {lm.lean})"
    )


def _forecast_block(fc: ScenarioForecast) -> list[str]:
    lines = [f"#### Horizon: {fc.horizon_days} trading days (`{fc.model_name}`)", ""]
    lines.append("| Scenario | Probability |")
    lines.append("| --- | --- |")
    for bucket in fc.buckets:
        lines.append(f"| {bucket.label} | {bucket.probability:.1%} |")
    lines.append("")
    lines.append(f"- Expected return: **{fc.expected_return:+.2%}**")
    lines.append(f"- P(up): {fc.upside_prob:.0%} · P(down): {fc.downside_prob:.0%}")
    big_move = _big_move_line(fc)
    if big_move is not None:
        lines.append(big_move)
    if fc.var_95 is not None and fc.var_99 is not None:
        lines.append(f"- VaR: 95% **{fc.var_95:.2%}**, 99% **{fc.var_99:.2%}**")
    if fc.ci_low is not None and fc.ci_high is not None and fc.ci_level is not None:
        lines.append(f"- {fc.ci_level:.0%} interval: [{fc.ci_low:.2%}, {fc.ci_high:.2%}]")
    lines.append(f"- Calibration: {fc.calibration_status}")
    trust = _horizon_trust(fc.horizon_days)
    if trust is not None:
        lines.append(f"- _Trust: {trust}._")
    if fc.notes:
        lines.append(f"- Note: {fc.notes}")
    lines.append("")
    return lines


def render_markdown(report: ResearchReport) -> str:
    """Return the full Markdown report."""
    out: list[str] = [
        f"# Research Report — {report.ticker}",
        f"_As of {report.as_of.isoformat()}_",
        "",
        _DISCLAIMER,
        "",
        "## Executive Summary",
        report.executive_summary or "_No summary available._",
        "",
    ]
    if report.key_observations:
        out += ["**Key observations**", *_bullets(report.key_observations), ""]
    if report.confidence_notes:
        out += [f"_{report.confidence_notes}_", ""]

    # Integrated analysis (Role C): reconciles forecast with news/earnings/technicals
    if report.synthesis is not None:
        s = report.synthesis
        out += ["## Integrated Analysis"]
        if s.overview:
            out += [s.overview, ""]
        if s.alignments:
            out += ["**Where signals align**", *_bullets(s.alignments), ""]
        if s.tensions:
            out += ["**Tensions / what tempers the forecast**", *_bullets(s.tensions), ""]
        if s.confidence:
            out += [f"_Confidence: {s.confidence}_", ""]

    # Technical analysis
    out += ["## Technical Analysis", report.technical_analysis.narrative or "_n/a_", ""]
    if report.technical_analysis.indicators:
        out += ["| Indicator | Value |", "| --- | --- |"]
        for key, value in report.technical_analysis.indicators.items():
            out.append(f"| {_LABELS.get(key, key)} | {_fmt_indicator(key, value)} |")
        out.append("")

    # Earnings proximity (context the price-only forecast can't see)
    if report.earnings is not None and report.earnings.next_earnings_date is not None:
        e = report.earnings
        line = f"**Next earnings:** {e.next_earnings_date} ({e.days_to_next_earnings} days)"
        if e.earnings_in_horizon:
            line += " — **inside the forecast horizon** (expect wider outcomes)"
        out += ["## Earnings", line, ""]

    # Probabilistic scenarios
    out += ["## Probabilistic Scenarios", ""]
    if report.forecasts:
        out.append(
            "_Model-generated (statistical baseline + calibrated ML where available); "
            "the LLM does not produce these numbers._"
        )
        out.append("")
        for fc in report.forecasts:
            out += _forecast_block(fc)
    else:
        out += ["_No forecasts available._", ""]

    # News analysis
    news = report.news_analysis
    out += ["## News Analysis", ""]
    out += ["**Recent developments**", *_bullets(news.recent_developments), ""]
    out += ["**Positive themes**", *_bullets(news.positive_themes), ""]
    out += ["**Negative themes**", *_bullets(news.negative_themes), ""]
    out += ["**Emerging risks**", *_bullets(news.emerging_risks), ""]

    # Fundamentals (optional)
    if report.fundamentals is not None:
        f = report.fundamentals
        out += ["## Fundamentals", ""]
        rows = {
            "Revenue": f.revenue,
            "Revenue growth (YoY)": f.revenue_growth_yoy,
            "EPS": f.eps,
            "P/E": f.pe_ratio,
            "Market cap": f.market_cap,
            "Profit margin": f.profit_margin,
        }
        out += ["| Metric | Value |", "| --- | --- |"]
        for label, amount in rows.items():
            if amount is not None:
                out.append(f"| {label} | {amount:,.4g} |")
        out.append("")

    # Structured signal lists
    out += ["## Bullish Factors", *_bullets(report.bullish_factors), ""]
    out += ["## Bearish Factors", *_bullets(report.bearish_factors), ""]
    out += ["## Risk Flags", *_bullets(report.risk_flags), ""]

    # Uncertainty
    out += ["## Uncertainty Notes", *_bullets(report.uncertainty_notes), ""]

    # Citations
    out += ["## Source Citations", ""]
    if report.citations:
        for c in report.citations:
            out.append(f"- [{c.title}]({c.url}) — {c.source}")
    else:
        out.append("- _No sources cited._")
    out.append("")

    return "\n".join(out)
