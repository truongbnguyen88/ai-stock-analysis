"""Golden-value tests for volume / liquidity indicators.

relative_volume(window=2) hand-derivation, volume=[10,20,30,40]:
  baseline_t = mean of the *prior* `window` bars (shift(1)):
    t0,t1 NaN (window not full on the shifted series);
    t2 = mean(10,20)=15 → 30/15 = 2.0;
    t3 = mean(20,30)=25 → 40/25 = 1.6
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stock_agent.indicators.volume import dollar_volume_zscore, relative_volume


def test_relative_volume_excludes_current_bar() -> None:
    vol = pd.Series([10.0, 20.0, 30.0, 40.0])
    out = relative_volume(vol, window=2).tolist()
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(1.6)


def test_relative_volume_is_one_on_flat_volume() -> None:
    out = relative_volume(pd.Series([100.0] * 10), window=3).tolist()
    # Constant volume → ratio against own trailing mean == 1 after warmup.
    assert out[-1] == pytest.approx(1.0)


def test_dollar_volume_zscore_scale_free_and_trailing() -> None:
    # close=1 so dollar_vol == volume. volume=[10,10,10,10,40], window=4.
    # At t4 prior 4 bars all 10 → mean=10, std=0 → z = (40-10)/0 = inf? std of
    # constant is 0; guard by using a non-degenerate prior.
    close = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    vol = pd.Series([10.0, 12.0, 14.0, 16.0, 100.0])
    out = dollar_volume_zscore(close, vol, window=4).tolist()
    # prior 4 dollar-vols at t4 = [10,12,14,16]: mean=13, std(ddof=1)=sqrt(20/3)
    mean, std = 13.0, math.sqrt(20.0 / 3.0)
    assert out[4] == pytest.approx((100.0 - mean) / std)
    # warmup: first `window` rows have no full prior window → NaN
    assert all(math.isnan(x) for x in out[:4])


def test_dollar_volume_zscore_uses_notional_not_shares() -> None:
    # Identical share volume; the only difference is price on the LAST bar (prior
    # 5 bars share price 10 in both, so their trailing stats are identical and
    # non-degenerate). Doubling last-bar price doubles its notional → larger z.
    vol = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0, 100.0])
    flat_price = pd.Series([10.0] * 6)
    spike_price = pd.Series([10.0] * 5 + [20.0])
    z_flat = dollar_volume_zscore(flat_price, vol, window=5).iloc[-1]
    z_spike = dollar_volume_zscore(spike_price, vol, window=5).iloc[-1]
    assert z_spike > z_flat  # price-driven notional surge detected
