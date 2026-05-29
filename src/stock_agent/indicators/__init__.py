"""Technical indicators — pure, vectorized functions over price data.

Per-indicator modules (returns/trend/momentum/volatility) operate on pandas
Series; ``frame`` adapts a ``PriceSeries`` to a DataFrame and ``snapshot``
assembles the latest values into a typed ``IndicatorSnapshot``.
"""

from stock_agent.indicators.frame import adjusted_close, to_ohlcv_frame
from stock_agent.indicators.momentum import ema, macd, rsi
from stock_agent.indicators.returns import cumulative_returns, daily_returns, log_returns
from stock_agent.indicators.snapshot import IndicatorSnapshot, compute_snapshot
from stock_agent.indicators.trend import moving_averages, sma
from stock_agent.indicators.volatility import (
    atr,
    drawdown_series,
    historical_volatility,
    max_drawdown,
    true_range,
)

__all__ = [
    # frame adapter
    "to_ohlcv_frame",
    "adjusted_close",
    # returns
    "daily_returns",
    "log_returns",
    "cumulative_returns",
    # trend
    "sma",
    "moving_averages",
    # momentum
    "rsi",
    "macd",
    "ema",
    # volatility / risk
    "historical_volatility",
    "true_range",
    "atr",
    "drawdown_series",
    "max_drawdown",
    # snapshot
    "IndicatorSnapshot",
    "compute_snapshot",
]
