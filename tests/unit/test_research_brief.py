"""Deterministic rich-brief renderer (research.brief.render_research_brief).

The brief is what the chat UI presents for research_summary — tables from the numbers + the memo's
grounded prose. These tests pin the structure (## sections → brass in the web app), table layout,
the deterministic signal descriptors, GFM pipe-escaping, and graceful omission of empty sections.
"""

from __future__ import annotations

from datetime import date

from stock_agent.research.brief import render_research_brief
from stock_agent.schemas.earnings import EarningsContext
from stock_agent.schemas.forecast import LargeMoveBreakdown, ProbBucket, ScenarioForecast
from stock_agent.schemas.research import NewsAnalysis, PriceSnapshot, ResearchMemo, SourceCitation

_D = date(2026, 7, 16)


def _fc(h: int, up: float, er: float, var: float, lo: float, hi: float) -> ScenarioForecast:
    return ScenarioForecast(
        ticker="MRVL", as_of=_D, horizon_days=h, model_name="ensemble",
        buckets=[
            ProbBucket(label="down", lower=None, upper=0.0, probability=round(1 - up, 4)),
            ProbBucket(label="up", lower=0.0, upper=None, probability=round(up, 4)),
        ],
        expected_return=er, upside_prob=up, downside_prob=round(1 - up, 4), var_95=var,
        ci_level=0.90, ci_low=lo, ci_high=hi,
    )


def _full_memo() -> ResearchMemo:
    return ResearchMemo(
        ticker="MRVL", as_of=_D,
        technical_indicators={
            "last_close": 191.36, "ma20": 256.60, "ma50": 235.16, "ma200": 129.26,
            "rsi14": 36.15, "macd": -10.86, "macd_signal": -0.76, "macd_hist": -10.10,
            "hist_vol_annualized": 0.933, "atr14": 22.64, "max_drawdown": -0.395,
            "last_daily_return": -0.072,
        },
        forecasts=[
            _fc(60, 0.674, 0.305, -0.458, -0.458, 2.14),
            _fc(20, 0.603, 0.078, -0.245, -0.245, 0.40),
        ],
        price_snapshot=PriceSnapshot(
            window_days=30, n_bars=21, first_close=278.0, last_close=191.36,
            period_high=310.50, period_low=191.36, pct_change=-0.313, last_return=-0.072,
        ),
        large_move=LargeMoveBreakdown(
            ticker="MRVL", as_of=_D, horizon_days=20, model_name="ml_logistic",
            threshold=0.05, prob_large_move=0.817, prob_big_up=0.513, prob_big_down=0.304,
            lean="up",
        ),
        earnings=EarningsContext(
            ticker="MRVL", as_of=_D, next_earnings_date=date(2026, 8, 27),
            last_earnings_date=date(2026, 5, 29), days_to_next_earnings=42,
            days_since_last_earnings=48, horizon_days=20, earnings_in_horizon=False,
        ),
        executive_summary="Marvell faces a sharp near-term sell-off even as models lean up.",
        business_drivers=["Custom ASIC demand for AI data center is the growth engine [1]."],
        risk_factors=["Customer concentration: a few customers drive a large share [6][9]."],
        bullish_evidence=["Record quarterly revenue of $2.418B, up 27.6% YoY."],
        bearish_evidence=["Stock declined 31.3% over 30 days; RSI at 36."],
        uncertainty_notes=["Next earnings 2026-08-27 falls outside the 20-day window."],
        news=NewsAnalysis(
            lookback_days=21, article_count=25, overview="AI demand narrative dominant.",
            bullish=["Record revenue $2.418B (+27.6% YoY)."], bearish=["Erste Group downgraded."],
            catalysts=["Hyperscaler capex."], risks=["China export restrictions."],
            pct_positive=0.24, pct_negative=0.08, avg_sentiment=0.16, sentiment_coverage=0.5,
        ),
        citations=[SourceCitation(marker=1, chunk_id="c1", label="MRVL 10-K Jan 2026 — Item 1")],
    )


# ---- structure: numbered brass sections in order ----
def test_brief_has_all_numbered_sections_in_order() -> None:
    md = render_research_brief(_full_memo())
    heads = [
        "# 🔬 MRVL — Executive Research Brief",
        "## 📌 Executive Summary",
        "## 📊 1. Price Snapshot",
        "## 📈 2. Technical Indicators",
        "## 🔮 3. Model Forecasts (Ensemble)",
        "## 💥 4. Large-Move Probability",
        "## 📅 5. Earnings Context",
        "## 📰 6. News & Sentiment",
        "## 🏢 7. Business Drivers & Risk Factors",
        "## ⚖️ 8. Bull vs Bear",
        "## 🔑 9. Key Uncertainties",
        "## 🗂 Source Citations",
    ]
    positions = [md.find(h) for h in heads]
    missing = [h for h, p in zip(heads, positions, strict=True) if p < 0]
    assert not missing, missing
    assert positions == sorted(positions)  # sections appear in the intended order
    assert "not financial advice" in md.lower()


# ---- tables: technicals + ensemble (the explicit asks) ----
def test_technical_indicators_render_as_a_signalled_table() -> None:
    md = render_research_brief(_full_memo())
    assert "| Indicator | Value | Signal |" in md
    assert "| MA20 | $256.60 | price below — bearish |" in md  # 191.36 < 256.60
    assert "| MA200 | $129.26 | price above — bullish |" in md  # 191.36 > 129.26
    assert "| RSI (14) | 36.15 | soft (below midpoint) |" in md
    assert "| MACD | -10.86 | below signal — bearish |" in md  # -10.86 < -0.76
    assert "| Annualized volatility | 93.3% | elevated |" in md
    assert "| Trend | **Sideways** | long-term structure |" in md


def test_ensemble_forecasts_render_as_a_multi_horizon_table() -> None:
    md = render_research_brief(_full_memo())
    assert "| Horizon | P(up) | Expected return | VaR 95% | 90% CI |" in md
    # Sorted ascending by horizon regardless of input order.
    i20, i60 = md.find("| 20 days |"), md.find("| 60 days |")
    assert 0 <= i20 < i60
    assert "| 20 days | 60.3% | +7.8% | -24.5% | [-24.5%, +40.0%] |" in md
    assert "| 60 days | 67.4% | +30.5% | -45.8% | [-45.8%, +214.0%] |" in md


def test_large_move_escapes_pipes_in_cell() -> None:
    # `|return|` must be escaped so GFM does not read the bars as column separators.
    md = render_research_brief(_full_memo())
    assert r"| P(\|return\| > 5%) | **82%** |" in md


# ---- bullets for the qualitative sections (the explicit asks) ----
def test_evidence_and_uncertainty_are_bullets() -> None:
    md = render_research_brief(_full_memo())
    assert "### 🟢 Bullish evidence\n- Record quarterly revenue of $2.418B, up 27.6% YoY." in md
    assert "### 🔴 Bearish evidence\n- Stock declined 31.3% over 30 days; RSI at 36." in md
    assert "- Next earnings 2026-08-27 falls outside the 20-day window." in md  # uncertainty
    assert "- Custom ASIC demand for AI data center is the growth engine [1]." in md  # driver+cite


# ---- graceful omission: no data → no empty tables ----
def test_sparse_memo_omits_empty_sections() -> None:
    memo = ResearchMemo(ticker="ABC", as_of=_D, executive_summary="Only a summary is available.")
    md = render_research_brief(memo)
    assert "# 🔬 ABC — Executive Research Brief" in md
    assert "## 📌 Executive Summary" in md
    assert "Only a summary is available." in md
    # None of the quantitative/qualitative sections appear (no data → omitted, no empty tables).
    for absent in ("Price Snapshot", "Technical Indicators", "Model Forecasts", "Large-Move",
                   "| Metric | Value |", "Earnings Context", "News & Sentiment", "Bull vs Bear",
                   "Key Uncertainties", "Source Citations"):
        assert absent not in md
    assert "not financial advice" in md.lower()
