"""Point-in-time price feature matrix.

All features at row t use only price data up to and including t.
This is guaranteed by construction: every operation is a rolling window
or a shift that looks backward, never forward. The assembler adds an
explicit leakage assertion as a belt-and-suspenders check.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date as Date

import pandas as pd

from stock_agent.data.earnings import days_to_next_earnings_series
from stock_agent.indicators.frame import adjusted_close, to_ohlcv_frame
from stock_agent.indicators.momentum import macd, rsi
from stock_agent.indicators.returns import daily_returns, intraday_return, overnight_return
from stock_agent.indicators.trend import moving_averages, pct_from_high
from stock_agent.indicators.volatility import (
    atr,
    bollinger_percent_b,
    drawdown_series,
    historical_volatility,
    realized_skewness,
    semivol_ratio,
)
from stock_agent.indicators.volume import dollar_volume_zscore, relative_volume
from stock_agent.schemas.market import PriceSeries

# Feature columns expected by the ML models. Keep consistent across train / infer.
# All scale-free (ratios / bounded indicators) so cross-ticker pooling is valid.
PRICE_FEATURE_COLS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "rsi14",
    "macd_hist",
    "price_to_ma20",
    "price_to_ma50",
    "ma50_to_ma200",
    "hist_vol_20",
    "hist_vol_60",
    "vol_ratio",
    "atr_pct",
    "drawdown",
    "B_perc",  # Bollinger %B: position within the 20-day volatility band
    # Market-wide volatility regime (VIX) — sharpens the big-move/vol signal:
    "vix_level",  # VIX / 100 (annualized vol fraction; NaN if unavailable)
    "vix_rel",  # VIX vs its own 20-day average (>1 = market vol rising)
    "days_to_next_earnings",  # leakage-safe cadence estimate (NaN if no earnings data)
    # --- Promoted candidate features (Phase 1.6 ablation; OHLCV-only, no extra fetch) ---
    # Walk-forward logistic ablation across h20/30/60 promoted these (Brier/ECE/AUC):
    # `shape` (robust winner, all metrics × all horizons), `volume` (clean low-risk add),
    # `session` (marginal but net-positive ECE/AUC). high52w/relstr were rejected (hurt
    # calibration). See validations_results.md.
    "rvol_20",  # volume vs its 20d trailing mean (participation surge)
    "dollar_vol_z_20",  # dollar-volume z-score vs 20d trailing distribution (liquidity surprise)
    "overnight_ret_20d",  # 20d mean overnight (close→open) return
    "intraday_ret_20d",  # 20d mean intraday (open→close) return
    "realized_skew_60",  # 60d realized skewness of log returns (tail asymmetry)
    "semivol_ratio_60",  # 60d downside/upside semideviation ratio
]

# --- Candidate feature groups (opt-in, ablation-gated) -----------------------
# Remaining UNPROMOTED candidates — computed only when explicitly requested via
# ``feature_groups`` so the default matrix (and committed artifacts) is unchanged.
# `high52w`/`relstr` were ablation-REJECTED (hurt calibration) but kept as opt-in
# infra; `insider` is not yet evaluated (its ablation was blocked by SEC throttling).
# Every column is scale-free + point-in-time. ``relstr`` needs a market index (SPY)
# via ``market=``; ``insider`` needs a Form 4 activity frame via ``insider=``.
FEATURE_GROUPS: dict[str, list[str]] = {
    "high52w": ["pct_from_52w_high"],  # ablation-rejected; retained as opt-in
    "relstr": ["rel_strength_20d", "rel_strength_60d"],  # rejected; requires market= (SPY)
    # Re-engineered insider signal (requires insider= Form 4 frame): separates the
    # informative BUY channel from sells, weights by Δ-ownership conviction + senior role.
    "insider": [
        "insider_buy_conviction_63d",
        "insider_senior_buy_63d",
        "insider_sell_pressure_63d",
    ],
}

# Trailing window (trading days) for insider aggregation: insider signals are slow,
# ~one quarter is the conventional accumulation horizon (PEAD/insider literature).
_INSIDER_WINDOW = 63
_INSIDER_MIN_PERIODS = 21


def resolve_feature_cols(feature_groups: Sequence[str] | None) -> list[str]:
    """Ordered column list for a feature-group selection (baseline + requested).

    ``None`` → baseline only (current production behavior). Unknown group names
    raise to fail loud rather than silently drop a requested signal.
    """
    cols = list(PRICE_FEATURE_COLS)
    for group in feature_groups or []:
        if group not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group {group!r}; valid: {sorted(FEATURE_GROUPS)}")
        cols.extend(FEATURE_GROUPS[group])
    return cols


def groups_for_cols(feature_cols: Sequence[str]) -> list[str]:
    """Infer which candidate groups a (trained-model) column list contains.

    Inference reads this off the artifact's ``feature_cols`` so the live feature
    vector is built with exactly the groups the model was trained on — no extra
    metadata needed. A group counts as present only if ALL its columns appear.
    """
    present = set(feature_cols)
    return [g for g, cols in FEATURE_GROUPS.items() if set(cols) <= present]


def build_price_feature_matrix(
    series: PriceSeries,
    *,
    earnings_dates: list[Date] | None = None,
    vix: pd.Series | None = None,
    market: pd.Series | None = None,
    insider: pd.DataFrame | None = None,
    feature_groups: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a date-indexed DataFrame of model features (one row per bar).

    Mostly price-derived; ``days_to_next_earnings`` is an earnings-cadence feature
    that is NaN unless ``earnings_dates`` is supplied. NaN is correct for rows
    where a lookback window is not yet full (e.g. MA200 on a 100-bar series).
    LightGBM handles NaN natively; scikit-learn models (logistic) receive imputed values.

    ``feature_groups`` opts in to candidate columns from ``FEATURE_GROUPS`` (default
    ``None`` → baseline only, i.e. unchanged production behavior). ``market`` is a
    market-index close series (SPY) used only by the ``relstr`` group; ``insider`` is
    a per-``filing_date`` Form 4 activity frame (``data.insider``) used only by the
    ``insider`` group.
    """
    frame = to_ohlcv_frame(series)
    close = adjusted_close(frame)
    cols = resolve_feature_cols(feature_groups)

    df = pd.DataFrame(index=frame.index)

    # Trailing returns at 1/5/20/60-day horizons.
    df["ret_1d"] = daily_returns(close)
    df["ret_5d"] = close / close.shift(5) - 1
    df["ret_20d"] = close / close.shift(20) - 1
    df["ret_60d"] = close / close.shift(60) - 1

    # Momentum indicators.
    df["rsi14"] = rsi(close)
    df["macd_hist"] = macd(close)["histogram"]

    # Price deviation from moving averages (signed percentage above/below MA).
    mas = moving_averages(close)
    df["price_to_ma20"] = close / mas["ma20"] - 1
    df["price_to_ma50"] = close / mas["ma50"] - 1
    # MA50 vs MA200: positive = golden cross (uptrend), negative = death cross.
    df["ma50_to_ma200"] = mas["ma50"] / mas["ma200"] - 1

    # Volatility regime features.
    df["hist_vol_20"] = historical_volatility(close, window=20)
    df["hist_vol_60"] = historical_volatility(close, window=60)
    # Vol ratio: > 1 = vol expanding, < 1 = vol compressing.
    df["vol_ratio"] = df["hist_vol_20"] / df["hist_vol_60"]

    # ATR as a fraction of price: normalized daily range.
    df["atr_pct"] = atr(frame["high"], frame["low"], frame["close"]) / close

    # Drawdown from the rolling peak (always <= 0).
    df["drawdown"] = drawdown_series(close)

    # Bollinger %B: where price sits within its 20-day volatility band (mean-reversion).
    df["B_perc"] = bollinger_percent_b(close, window=20)

    # Market-wide volatility regime (VIX). Same for every ticker on a date, so it mainly
    # sharpens the volatility / big-move signal. Point-in-time safe: VIX[t] is known at t;
    # vix_rel uses a backward rolling window. NaN when VIX is unavailable (handled like any
    # missing feature). Aligned to the price dates by ffill (uses VIX at-or-before each date).
    if vix is not None and not vix.empty:
        vix_level = vix / 100.0  # annualized vol fraction, comparable to hist_vol_*
        vix_rel = vix / vix.rolling(window=20, min_periods=20).mean()
        df["vix_level"] = vix_level.reindex(df.index, method="ffill")
        df["vix_rel"] = vix_rel.reindex(df.index, method="ffill")
    else:
        df["vix_level"] = float("nan")
        df["vix_rel"] = float("nan")

    # Earnings proximity (leakage-safe cadence estimate; same calc at train & infer).
    feat_dates = [pd.Timestamp(idx).date() for idx in df.index]
    df["days_to_next_earnings"] = pd.Series(
        days_to_next_earnings_series(feat_dates, earnings_dates or []),
        index=df.index,
        dtype="float64",
    )

    # --- Promoted candidate features (Phase 1.6; OHLCV-only, always computed) ---
    # Volume normalized against its own trailing history (scale-free).
    df["rvol_20"] = relative_volume(frame["volume"], window=20)
    df["dollar_vol_z_20"] = dollar_volume_zscore(close, frame["volume"], window=20)
    # Decompose the daily move into overnight (close→open) and intraday (open→close)
    # drift; 20-day trailing means smooth the per-bar noise. Scale-free returns.
    df["overnight_ret_20d"] = (
        overnight_return(frame["open"], close).rolling(window=20, min_periods=20).mean()
    )
    df["intraday_ret_20d"] = (
        intraday_return(frame["open"], close).rolling(window=20, min_periods=20).mean()
    )
    # Return-distribution shape: skew + downside/upside semideviation ratio.
    df["realized_skew_60"] = realized_skewness(close, window=60)
    df["semivol_ratio_60"] = semivol_ratio(close, window=60)

    # --- Remaining candidate feature groups (computed only when requested) ----
    requested = set(feature_groups or [])
    if "high52w" in requested:
        # Nearness-to-52w-high anchoring momentum (<= 0; 0 = at the high).
        df["pct_from_52w_high"] = pct_from_high(close, window=252, min_periods=20)
    if "relstr" in requested:
        # Market-relative (residual) momentum: stock trailing return minus the market's,
        # which strips out beta-driven moves. NaN if no market series supplied. The market
        # return is computed on its native calendar then aligned to price dates by ffill.
        if market is not None and not market.empty:
            m_ret20 = (market / market.shift(20) - 1.0).reindex(df.index, method="ffill")
            m_ret60 = (market / market.shift(60) - 1.0).reindex(df.index, method="ffill")
            df["rel_strength_20d"] = df["ret_20d"] - m_ret20
            df["rel_strength_60d"] = df["ret_60d"] - m_ret60
        else:
            df["rel_strength_20d"] = float("nan")
            df["rel_strength_60d"] = float("nan")
    if "insider" in requested:
        # Re-engineered insider (Form 4) signal — separate buy/sell channels, weighted
        # by Δ-ownership conviction + senior role, 10b5-1 filtered (see data/insider.py):
        #   insider_buy_conviction_63d = trailing-63d Σ opportunistic-buy Δ-ownership
        #   insider_senior_buy_63d     = trailing-63d count of CEO/CFO opportunistic buys
        #   insider_sell_pressure_63d  = trailing-63d Σ opportunistic-sell |Δ-ownership|
        # Activity is keyed by filing_date (public date) and reindexed to price dates
        # (0 = no filing), so the trailing rolling sums only ever see past filings.
        if insider is not None and not insider.empty:
            aligned = insider.reindex(df.index, fill_value=0.0)

            def _roll(s: pd.Series) -> pd.Series:
                return s.rolling(_INSIDER_WINDOW, min_periods=_INSIDER_MIN_PERIODS).sum()

            df["insider_buy_conviction_63d"] = _roll(aligned["buy_conviction"])
            df["insider_senior_buy_63d"] = _roll(aligned["senior_buy_n"])
            df["insider_sell_pressure_63d"] = _roll(aligned["sell_pressure"])
        else:
            df["insider_buy_conviction_63d"] = float("nan")
            df["insider_senior_buy_63d"] = float("nan")
            df["insider_sell_pressure_63d"] = float("nan")

    return df[cols]


def current_price_features(series: PriceSeries) -> dict[str, float | None]:
    """Return the most recent row of the feature matrix as a dict."""
    df = build_price_feature_matrix(series)
    last = df.iloc[-1]
    return {col: (float(last[col]) if pd.notna(last[col]) else None) for col in PRICE_FEATURE_COLS}
