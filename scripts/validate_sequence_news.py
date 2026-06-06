"""Does a DEEP, TUNED sequence model extract news signal tabular ML couldn't? (Task 8 × 10)

Stronger retry than the shallow spike. For each arm (price-only vs price+news) it runs the
careful tuning protocol from `sequence_tune` — a DEEP-LSTM grid (3–4 layers, wide hidden,
LayerNorm), early stopping on validation Brier, temperature calibration — picks the best
config by calibrated val Brier, then scores it on a **held-out TEST split it never saw**.

Fair ablation: both arms use identical windows over full history (news features are NaN→0
before GDELT coverage, real after) and identical train/val/test dates; they differ ONLY by
the news channel. Train spans full history (max data for the deep net) but includes ~2yr of
news; val/test are recent and fully news-covered. The decision metric is **TEST** big-move
AUC at the INNER band (h20 |r|>0.05, h60 |r|>0.15 — `thresholds[3]`, where there's real skill
and enough events) + TEST exceedance Brier. The train-vs-val AUC gap flags overfit.

Run: PYTHONPATH=src python scripts/validate_sequence_news.py
"""

from __future__ import annotations

import warnings
from datetime import date

from stock_agent.features.news_history import load_news_store
from stock_agent.forecasting.buckets import thresholds_for_horizon
from stock_agent.forecasting.sequence_tune import (
    _assemble_split,
    _big_move_auc,
    _brier,
    _predict_probs,
    _train_config,
    deep_config_grid,
)
from stock_agent.forecasting.train_pooled import fetch_universe_series
from stock_agent.providers.registry import build_default_registry
from stock_agent.settings import get_settings

warnings.filterwarnings("ignore")

BASKET = [
    "NVDA", "AMD", "AVGO", "MU", "MRVL", "TSLA", "ARM", "PLTR", "SMCI", "MPWR",  # semis/AI
    "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "ORCL", "CRM", "ADBE", "CSCO",  # tech
    "JPM", "GS", "BAC", "V", "MA",  # financials
    "UNH", "LLY", "JNJ", "PFE", "ABBV",  # healthcare
    "CAT", "BA", "WMT", "HD", "KO", "XOM", "CVX",  # industrials/consumer/energy
]
HORIZONS = [20, 60]
TRAIN_END = date(2025, 1, 1)  # train = full history ≤ here (incl. ~2yr news)
VAL_END = date(2025, 7, 1)  # val = (TRAIN_END, VAL_END]; test = (VAL_END, end] — news-covered
STRIDE = 3  # subsample overlapping windows to bound the deep nets' compute
SEED = 42


def _tune_arm(series, *, horizon, thresholds, news_store):  # type: ignore[no-untyped-def]
    """Tune the deep grid for one arm; return (best_model, val_brier, val_auc, train_auc, split)."""
    configs = deep_config_grid()
    splits = {}  # per-lookback (the grid mixes lookback 60/90)
    best = None
    for cfg in configs:
        if cfg.lookback not in splits:
            splits[cfg.lookback] = _assemble_split(
                series, horizon=horizon, lookback=cfg.lookback, train_end=TRAIN_END,
                val_end=VAL_END, stride=STRIDE, news_store=news_store,
            )
        split = splits[cfg.lookback]
        model, vb, vauc, tauc = _train_config(
            split, cfg, horizon=horizon, thresholds=thresholds, seed=SEED
        )
        print(f"    h{cfg.hidden} L{cfg.layers} lb{cfg.lookback} do{cfg.dropout} "
              f"T{model.temperature:.2f} -> valBrier {vb:.4f}  valAUC "
              f"{vauc if vauc is None else round(vauc, 3)}  trainAUC "
              f"{tauc if tauc is None else round(tauc, 3)}", flush=True)
        if best is None or vb < best[1]:
            best = (model, vb, vauc, tauc, split)
    assert best is not None
    return best


def main() -> None:
    series = fetch_universe_series(BASKET, build_default_registry(get_settings()))
    store = load_news_store()
    print(f"universe: {len(series)} tickers | news present: "
          f"{len(store.tickers() & {s.ticker for s in series})} | "
          f"train≤{TRAIN_END}  val≤{VAL_END}  test>{VAL_END}")

    for horizon in HORIZONS:
        thresholds = thresholds_for_horizon(horizon)
        k_big = thresholds[3]  # inner big-move band: 0.05 (h20) / 0.15 (h60)
        print(f"\n{'=' * 78}\nHORIZON {horizon}  (big-move band |r|>{k_big:.2f})\n{'=' * 78}")
        out = {}
        for arm, ns in (("price", None), ("price+news", store)):
            print(f"\n  [{arm}]  tuning deep grid …")
            model, vb, vauc, tauc, split = _tune_arm(
                series, horizon=horizon, thresholds=thresholds, news_store=ns
            )
            te_probs = _predict_probs(model, split.Xte)
            te_brier = _brier(te_probs, split.fte, thresholds)
            te_auc = _big_move_auc(te_probs, split.fte, k_big)
            out[arm] = dict(val_brier=vb, val_auc=vauc, train_auc=tauc,
                            test_brier=te_brier, test_auc=te_auc, n_test=len(split.fte),
                            n_feat=split.n_features)
            print(f"    BEST: valBrier {vb:.4f} | TEST Brier {te_brier:.4f}  "
                  f"big-move AUC {te_auc if te_auc is None else round(te_auc, 3)}  "
                  f"(n_test={len(split.fte)}, feat={split.n_features})")

        p, n = out["price"], out["price+news"]
        d_auc = (n["test_auc"] or 0) - (p["test_auc"] or 0)
        d_bri = n["test_brier"] - p["test_brier"]
        print(f"\n  {'arm':>11} | {'TEST Brier':>10} | {'TEST bigAUC':>11} | {'val Brier':>9}")
        for arm in ("price", "price+news"):
            o = out[arm]
            print(f"  {arm:>11} | {o['test_brier']:>10.4f} | "
                  f"{(o['test_auc'] or float('nan')):>11.3f} | {o['val_brier']:>9.4f}")
        helps = d_auc > 0 and d_bri < 0
        print(f"\n  Δ(news−price): TEST bigAUC {d_auc:+.3f}, TEST Brier {d_bri:+.4f}")
        print(f"  → news {'HELPS' if helps else 'does NOT help'} the deep LSTM at h{horizon}")


if __name__ == "__main__":
    main()
