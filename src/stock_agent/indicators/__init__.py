"""Technical indicators — pure, vectorized functions over price data.

Per-indicator modules (returns/trend/momentum/volatility) operate on pandas
Series; ``frame`` adapts a ``PriceSeries`` to a DataFrame and ``snapshot``
assembles the latest values into a typed ``IndicatorSnapshot``.
"""

from stock_agent.indicators.frame import adjusted_close, to_ohlcv_frame
from stock_agent.indicators.momentum import ema, macd, rsi
from stock_agent.indicators.returns import (
    cumulative_returns,
    daily_returns,
    intraday_return,
    log_returns,
    overnight_return,
)
from stock_agent.indicators.snapshot import IndicatorSnapshot, compute_snapshot
from stock_agent.indicators.trend import moving_averages, pct_from_high, sma
from stock_agent.indicators.volatility import (
    atr,
    bollinger_percent_b,
    drawdown_series,
    historical_volatility,
    max_drawdown,
    realized_skewness,
    semivol_ratio,
    true_range,
)
from stock_agent.indicators.volume import dollar_volume_zscore, relative_volume

__all__ = [
    # frame adapter
    "to_ohlcv_frame",
    "adjusted_close",
    # returns
    "daily_returns",
    "log_returns",
    "cumulative_returns",
    "overnight_return",
    "intraday_return",
    # trend
    "sma",
    "moving_averages",
    "pct_from_high",
    # momentum
    "rsi",
    "macd",
    "ema",
    # volatility / risk
    "historical_volatility",
    "true_range",
    "atr",
    "bollinger_percent_b",
    "drawdown_series",
    "max_drawdown",
    "realized_skewness",
    "semivol_ratio",
    # volume / liquidity
    "relative_volume",
    "dollar_volume_zscore",
    # snapshot
    "IndicatorSnapshot",
    "compute_snapshot",
]
