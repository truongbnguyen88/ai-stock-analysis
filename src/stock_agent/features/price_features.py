"""Point-in-time price feature matrix.

All features at row t use only price data up to and including t.
This is guaranteed by construction: every operation is a rolling window
or a shift that looks backward, never forward. The assembler adds an
explicit leakage assertion as a belt-and-suspenders check.
"""

from __future__ import annotations

import pandas as pd

from stock_agent.indicators.frame import adjusted_close, to_ohlcv_frame
from stock_agent.indicators.momentum import macd, rsi
from stock_agent.indicators.returns import daily_returns
from stock_agent.indicators.trend import moving_averages
from stock_agent.indicators.volatility import (
    atr,
    bollinger_percent_b,
    drawdown_series,
    historical_volatility,
)
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
]


def build_price_feature_matrix(series: PriceSeries) -> pd.DataFrame:
    """Return a date-indexed DataFrame of price features (one row per bar).

    NaN is correct for rows where a lookback window is not yet full (e.g. MA200
    on a 100-bar series). XGBoost handles NaN natively; scikit-learn models
    receive imputed values from the assembler.
    """
    frame = to_ohlcv_frame(series)
    close = adjusted_close(frame)

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

    return df[PRICE_FEATURE_COLS]


def current_price_features(series: PriceSeries) -> dict[str, float | None]:
    """Return the most recent row of the feature matrix as a dict."""
    df = build_price_feature_matrix(series)
    last = df.iloc[-1]
    return {col: (float(last[col]) if pd.notna(last[col]) else None) for col in PRICE_FEATURE_COLS}
