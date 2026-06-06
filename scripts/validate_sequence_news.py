"""Does a SEQUENCE model extract signal from news that tabular ML couldn't? (Task 8 × 10)

Two priors say no: (1) the LSTM didn't beat the tabular toolkit on price; (2) news didn't
beat price-only in pooled tabular ML. This tests the remaining cell: news sentiment is
*temporal* (spikes, decay, momentum) — a structure the one-day tabular snapshot flattens.
A pooled LSTM is the one class that could exploit it.

Design (fair, same-window ablation):
  - Both arms train on the SAME leakage-safe windows over full sliced history; they differ
    ONLY by the news channel (news features are NaN→0 before GDELT coverage, real after), so
    any gap is news's marginal value — not a data-volume artifact.
  - Walk-forward, embargoed; the test folds are all inside the news-covered period.
  - Headline targets mirror the tabular news A/B: direction (P(r>0)) and big-move
    (P(|r|>outer-band)). ΔBrier<0 / ΔAUC>0 = news helps the sequence model.

Bounded for a CPU run (single-threaded for determinism): ~16-ticker pool, 4 folds,
horizons {20,60}, small early-ish LSTM. Reference bar = the recorded pooled-ML numbers
(validations_results.md); if news doesn't beat price-only HERE, the model-class question is
settled negative.

Run: PYTHONPATH=src python scripts/validate_sequence_news.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from stock_agent.features.news_history import load_news_store
from stock_agent.forecasting.buckets import buckets_for_horizon
from stock_agent.forecasting.sequence import SequenceForecaster, train_sequence_model
from stock_agent.forecasting.train_pooled import fetch_universe_series
from stock_agent.providers.registry import build_default_registry
from stock_agent.schemas.market import PriceSeries
from stock_agent.settings import get_settings

warnings.filterwarnings("ignore")

BASKET = [
    "NVDA", "AMD", "AVGO", "MU", "TSLA", "PLTR",  # volatile semis / AI
    "AAPL", "MSFT", "AMZN", "META", "GOOGL", "ORCL",  # mega-cap tech
    "JPM", "UNH", "WMT", "XOM",  # financials / healthcare / consumer / energy
]
HORIZONS = [20, 60]
FOLD_STARTS = [pd.Timestamp(s) for s in ("2024-10-01", "2025-04-01", "2025-10-01", "2026-01-01")]
FOLD_LEN = pd.DateOffset(months=3)
TEST_STRIDE = 5  # subsample test as-ofs (h-day labels overlap anyway) to bound forecasts
# Small, lightly-trained net — the news-covered window is short; keep it from overfitting.
LSTM = dict(hidden=32, layers=1, epochs=12, lookback=60, dropout=0.1, seed=42)


def _embargo(horizon: int) -> pd.Timedelta:
    return pd.Timedelta(days=int(np.ceil(horizon * 1.5)))  # kill target overlap (trading→cal)


def _slice(series_list: list[PriceSeries], cutoff: pd.Timestamp) -> list[PriceSeries]:
    out = []
    for s in series_list:
        bars = [b for b in s.bars if pd.Timestamp(b.date) <= cutoff]
        if len(bars) >= 150:  # enough for lookback + some windows
            out.append(PriceSeries(ticker=s.ticker, bars=bars))
    return out


def _collect(fc_model, series: PriceSeries, *, horizon: int, test: pd.DatetimeIndex,
             k_outer: float) -> list[tuple[float, float, float, int, int]]:
    """Per strided as-of in the fold: (p_up, p_big, realized, label_up, label_big)."""
    dates = pd.DatetimeIndex([pd.Timestamp(d) for d in series.dates])
    closes = np.asarray(series.closes, dtype=float)
    rows: list[tuple[float, float, float, int, int]] = []
    idxs = [i for i, d in enumerate(dates) if d in test and i + horizon < len(dates)]
    for i in idxs[::TEST_STRIDE]:
        sub = PriceSeries(ticker=series.ticker, bars=series.bars[: i + 1])
        fc = fc_model.forecast(sub, horizon_days=horizon, as_of=dates[i].date())
        realized = float(closes[i + horizon] / closes[i] - 1.0)
        p_big = fc.buckets[0].probability + fc.buckets[-1].probability
        rows.append((fc.upside_prob, p_big, realized,
                     int(realized > 0), int(abs(realized) > k_outer)))
    return rows


def _score(rows: list[tuple[float, float, float, int, int]]) -> dict[str, float]:
    a = np.array(rows, dtype=float)
    p_up, p_big, _r, y_up, y_big = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    out = {
        "brier_dir": brier_score_loss(y_up, p_up),
        "brier_big": brier_score_loss(y_big, p_big),
    }
    out["auc_dir"] = roc_auc_score(y_up, p_up) if len(np.unique(y_up)) > 1 else np.nan
    out["auc_big"] = roc_auc_score(y_big, p_big) if len(np.unique(y_big)) > 1 else np.nan
    return out


def main() -> None:
    registry = build_default_registry(get_settings())
    series = fetch_universe_series(BASKET, registry)
    store = load_news_store()
    print(f"universe: {len(series)} tickers | news tickers present: "
          f"{len(store.tickers() & {s.ticker for s in series})}")

    for horizon in HORIZONS:
        k_outer = float(buckets_for_horizon(horizon)[-1][1] or 0.10)  # BucketDef=(label,lo,hi)
        print(f"\n{'=' * 76}\nHORIZON {horizon}  (big-move band |r|>{k_outer:.2f})\n{'=' * 76}")
        deltas: dict[str, list[float]] = {}
        abs_price: dict[str, list[float]] = {}
        abs_news: dict[str, list[float]] = {}
        for f0 in FOLD_STARTS:
            f1 = f0 + FOLD_LEN
            train_end = f0 - _embargo(horizon)
            sliced = _slice(series, train_end)
            if len(sliced) < 6:
                continue
            price_fc = SequenceForecaster(train_sequence_model(sliced, horizon=horizon, **LSTM))
            news_model = train_sequence_model(sliced, horizon=horizon, news_store=store, **LSTM)
            news_fc = SequenceForecaster(news_model, news_store=store)
            test = pd.date_range(f0, f1, freq="D")
            rp, rn = [], []
            for s in series:
                rp += _collect(price_fc, s, horizon=horizon, test=test, k_outer=k_outer)
                rn += _collect(news_fc, s, horizon=horizon, test=test, k_outer=k_outer)
            if len(rp) < 50:
                continue
            sp, sn = _score(rp), _score(rn)
            for m in sp:
                d = (sn[m] - sp[m]) * (-1 if m.startswith("brier") else 1)  # +ve = news better
                deltas.setdefault(m, []).append(d)
                abs_price.setdefault(m, []).append(sp[m])
                abs_news.setdefault(m, []).append(sn[m])
            print(f"  fold {f0.date()}  n={len(rp):4d} | "
                  f"big-move Brier {sp['brier_big']:.4f}→{sn['brier_big']:.4f}  "
                  f"AUC {sp['auc_big']:.3f}→{sn['auc_big']:.3f}")

        print(f"\n  {'metric':>10} | {'price':>7} {'+news':>7} | mean Δ (news better >0) | folds")
        for m in ("brier_dir", "brier_big", "auc_dir", "auc_big"):
            if m not in deltas:
                continue
            dd = np.array(deltas[m])
            print(f"  {m:>10} | {np.nanmean(abs_price[m]):7.4f} {np.nanmean(abs_news[m]):7.4f} | "
                  f"{np.nanmean(dd):+.4f}  (helps {np.mean(dd > 0):.0%}) | {len(dd)}")
        big = np.array(deltas.get("brier_big", [0.0]))
        verdict = "HELPS" if np.nanmean(big) > 0 and np.mean(big > 0) > 0.5 else "does NOT help"
        print(f"\n  → news {verdict} the LSTM at h{horizon} (big-move Brier)")


if __name__ == "__main__":
    main()
