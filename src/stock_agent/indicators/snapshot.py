"""High-level indicator snapshot.

Computes the full indicator set from a ``PriceSeries`` and extracts the latest
("as of") values into a typed ``IndicatorSnapshot``. This is what pipelines and
the report consume; the per-indicator modules remain the pure, testable core.

Values that need more history than is available are ``None`` (e.g. MA200 on a
60-bar series) — surfaced honestly rather than fabricated.
"""

from __future__ import annotations

from datetime import date as Date

import pandas as pd
from pydantic import BaseModel

from stock_agent.indicators.frame import adjusted_close, to_ohlcv_frame
from stock_agent.indicators.momentum import macd, rsi
from stock_agent.indicators.returns import daily_returns
from stock_agent.indicators.trend import moving_averages
from stock_agent.indicators.volatility import atr, drawdown_series, historical_volatility
from stock_agent.schemas.market import PriceSeries


class IndicatorSnapshot(BaseModel):
    """Latest indicator values for one ticker (None == insufficient history)."""

    as_of: Date
    last_close: float
    last_daily_return: float | None = None

    ma20: float | None = None
    ma50: float | None = None
    ma200: float | None = None

    rsi14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None

    hist_vol_annualized: float | None = None
    atr14: float | None = None
    max_drawdown: float = 0.0

    # Trend-regime flags (descriptive only, not signals).
    price_above_ma50: bool | None = None
    price_above_ma200: bool | None = None
    ma50_above_ma200: bool | None = None
    trend_label: str = "insufficient_data"

    def numeric_indicators(self) -> dict[str, float]:
        """Flatten non-null numeric indicators for the report's technical section."""
        fields = [
            "last_close",
            "last_daily_return",
            "ma20",
            "ma50",
            "ma200",
            "rsi14",
            "macd",
            "macd_signal",
            "macd_hist",
            "hist_vol_annualized",
            "atr14",
            "max_drawdown",
        ]
        out: dict[str, float] = {}
        for name in fields:
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value)
        return out


def _last(series: pd.Series) -> float | None:
    """Return the last value as a float, or None if absent/NaN."""
    if len(series) == 0:
        return None
    value = series.iloc[-1]
    return float(value) if pd.notna(value) else None


def _trend_label(last_close: float, ma50: float | None, ma200: float | None) -> str:
    """Heuristic regime descriptor from price vs MA50/MA200 (not a signal).

    - uptrend:   price > MA50 and (MA200 unknown or MA50 > MA200)
    - downtrend: price < MA50 and (MA200 unknown or MA50 < MA200)
    - sideways:  anything else with enough data
    """
    if ma50 is None:
        return "insufficient_data"
    if last_close > ma50 and (ma200 is None or ma50 > ma200):
        return "uptrend"
    if last_close < ma50 and (ma200 is None or ma50 < ma200):
        return "downtrend"
    return "sideways"


def compute_snapshot(series: PriceSeries) -> IndicatorSnapshot:
    """Compute all indicators and return the latest-value snapshot."""
    frame = to_ohlcv_frame(series)
    close = adjusted_close(frame)

    mas = moving_averages(close)
    macd_df = macd(close)

    ma20 = _last(mas["ma20"])
    ma50 = _last(mas["ma50"])
    ma200 = _last(mas["ma200"])
    last_close = float(close.iloc[-1])

    return IndicatorSnapshot(
        as_of=series.bars[-1].date,
        last_close=last_close,
        last_daily_return=_last(daily_returns(close)),
        ma20=ma20,
        ma50=ma50,
        ma200=ma200,
        rsi14=_last(rsi(close)),
        macd=_last(macd_df["macd"]),
        macd_signal=_last(macd_df["signal"]),
        macd_hist=_last(macd_df["histogram"]),
        hist_vol_annualized=_last(historical_volatility(close)),
        atr14=_last(atr(frame["high"], frame["low"], frame["close"])),
        max_drawdown=float(drawdown_series(close).min()),
        price_above_ma50=(last_close > ma50) if ma50 is not None else None,
        price_above_ma200=(last_close > ma200) if ma200 is not None else None,
        ma50_above_ma200=(ma50 > ma200) if (ma50 is not None and ma200 is not None) else None,
        trend_label=_trend_label(last_close, ma50, ma200),
    )
