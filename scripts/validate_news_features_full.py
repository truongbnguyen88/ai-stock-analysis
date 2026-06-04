"""FULL walk-forward OOS validation: do GDELT news features beat price-only? (Task 10)

Rigorous follow-up to the single-split A/B (`validate_news_features.py`). Adds:
  - **Walk-forward** expanding folds, embargoed by horizon (multiple OOS regimes).
  - **~30-ticker** basket; horizons **20 / 30 / 60**; **logistic AND lightgbm**.
  - **Directional + big-move** targets; **per-ticker win-rate** (does news help ANY
    subset?) alongside pooled deltas with dispersion across folds.
  - **log1p** on the heavy-tailed buzz ratios (the normalization gap).
  - A **lightgbm retune** on the 25-feature set, scored on a DISJOINT held-out fold
    (guards against the winner's curse) — gives news its best legitimate shot.

Both arms get identical rows / imputer / model at every fold; only the 7 news
columns differ. ΔBrier<0 or ΔAUC>0 = news helps.

Run: PYTHONPATH=src python scripts/validate_news_features_full.py
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score

from stock_agent.data.market_context import fetch_vix
from stock_agent.features.assembler import _target_col, build_training_matrix
from stock_agent.features.news_history import (
    NEWS_HISTORY_COLS,
    build_news_history_features,
    load_news_store,
)
from stock_agent.forecasting.buckets import thresholds_for_horizon
from stock_agent.forecasting.pooled import _make_classifier
from stock_agent.forecasting.train_pooled import fetch_universe_earnings, fetch_universe_series
from stock_agent.providers.registry import build_default_registry
from stock_agent.settings import get_settings

warnings.filterwarnings("ignore")

BASKET = [
    # volatile semis / AI
    "NVDA", "AMD", "AVGO", "MU", "MRVL", "SMCI", "TSLA", "ARM", "PLTR", "MPWR",
    # mega-cap tech
    "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "ORCL", "CRM",
    # financials / healthcare / industrials / consumer / energy (stable mix)
    "JPM", "GS", "BAC", "UNH", "LLY", "JNJ", "CAT", "BA", "WMT", "KO", "XOM", "CVX",
]
HORIZONS = [20, 30, 60]
NEWS_START = pd.Timestamp("2023-04-01")
BUZZ_COLS = ["news_buzz", "epu_buzz"]

# Walk-forward test windows (quarterly); train = rows before (test_start - embargo).
FOLD_STARTS = [pd.Timestamp(s) for s in (
    "2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01",
    "2025-07-01", "2025-10-01", "2026-01-01",
)]
FOLD_LEN = pd.DateOffset(months=3)


def build_pooled(horizon: int):
    """Return (X[25 cols], y[thresholds], meta[date,ticker]) pooled across the basket."""
    registry = build_default_registry(get_settings())
    series = fetch_universe_series(BASKET, registry)
    earnings = fetch_universe_earnings([s.ticker for s in series], registry)
    store = load_news_store()
    vix = fetch_vix(registry, start=date(2018, 1, 1), end=date.today())
    thresholds = thresholds_for_horizon(horizon)

    xs, ys, meta = [], [], []
    for s in series:
        try:
            X, y = build_training_matrix(
                s, horizon, earnings_dates=earnings.get(s.ticker.upper()),
                vix=vix, thresholds=thresholds,
            )
        except ValueError:
            continue
        news = build_news_history_features(s.ticker, pd.DatetimeIndex(X.index), store)
        news[BUZZ_COLS] = np.log1p(news[BUZZ_COLS])  # tame heavy-tailed buzz ratios
        Xn = pd.concat([X, news], axis=1)
        dates = pd.DatetimeIndex(Xn.index)
        mask = dates >= NEWS_START
        if mask.sum() == 0:
            continue
        xs.append(Xn.loc[mask])
        ys.append(y.loc[mask])
        meta.append(pd.DataFrame({"date": dates[mask], "ticker": s.ticker}))
    X_all = pd.concat(xs, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    y_all = pd.concat(ys, ignore_index=True)
    m_all = pd.concat(meta, ignore_index=True)
    return X_all, y_all, m_all, thresholds


def _embargo_days(horizon: int) -> int:
    return int(np.ceil(horizon * 1.5))  # trading→calendar cushion to kill target overlap


def _fit_predict(clf, Xtr, ytr, Xte):
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    clf.fit(imp.transform(Xtr), ytr)
    return clf.predict_proba(imp.transform(Xte))[:, 1]


def _scores(y_true, p):
    b = brier_score_loss(y_true, p)
    a = roc_auc_score(y_true, p) if y_true.nunique() > 1 else np.nan
    return b, a


def _targets(y_all, thresholds):
    """Yield (name, target_series) for directional thresholds + big-move bands."""
    for thr in thresholds:
        yield f"dir {thr:+.2f}", y_all[_target_col(thr)]
    inner = min(abs(t) for t in thresholds if t != 0)
    outer = max(abs(t) for t in thresholds)
    for k in (inner, outer):
        up, dn = y_all[_target_col(k)], y_all[_target_col(-k)]
        yield f"big |r|>{k:.2f}", ((up == 1) | (dn == 0)).astype(float)


def walk_forward(X_all, y_all, m_all, thresholds, model_type, horizon):
    """Return aggregated ΔBrier/ΔAUC over folds + per-ticker win-rate for one model."""
    price_cols = [c for c in X_all.columns if c not in NEWS_HISTORY_COLS]
    emb = pd.Timedelta(days=_embargo_days(horizon))
    d = pd.DatetimeIndex(m_all["date"])

    dbri, dauc = {}, {}  # target -> list of per-fold deltas
    tick_wins, tick_total = 0, 0
    for f0 in FOLD_STARTS:
        f1 = f0 + FOLD_LEN
        te = (d >= f0) & (d < f1)
        tr = d < (f0 - emb)
        if te.sum() < 50 or tr.sum() < 500:
            continue
        for name, ytarget in _targets(y_all, thresholds):
            ytr, yte = ytarget[tr], ytarget[te]
            if ytr.nunique() < 2 or yte.nunique() < 2:
                continue
            p_price = _fit_predict(_make_classifier(model_type), X_all.loc[tr, price_cols], ytr,
                                   X_all.loc[te, price_cols])
            p_news = _fit_predict(_make_classifier(model_type), X_all.loc[tr], ytr, X_all.loc[te])
            bp, ap = _scores(yte, p_price)
            bn, an = _scores(yte, p_news)
            dbri.setdefault(name, []).append(bn - bp)
            dauc.setdefault(name, []).append(an - ap)
            # per-ticker win-rate (only on the directional 0% + big-move outer to limit noise)
            if name.startswith("big") or name == "dir +0.00":
                tk = m_all.loc[te, "ticker"].to_numpy()
                yy_all = yte.to_numpy()
                for t in np.unique(tk):
                    sel = tk == t
                    yy = yy_all[sel]
                    if len(np.unique(yy)) < 2:
                        continue
                    if brier_score_loss(yy, p_news[sel]) < brier_score_loss(yy, p_price[sel]):
                        tick_wins += 1
                    tick_total += 1
    return dbri, dauc, (tick_wins, tick_total)


def retune_lightgbm(X_all, y_all, m_all, thresholds, horizon=20, n_configs=40, seed=0):
    """Random hyperparameter search on 25 features; selected on VAL, reported on TEST.

    Gives news its best shot without the winner's curse: tune on train→val, then the
    single reported number is on a disjoint final test fold.
    """
    rng = np.random.default_rng(seed)
    price_cols = [c for c in X_all.columns if c not in NEWS_HISTORY_COLS]
    d = pd.DatetimeIndex(m_all["date"])
    emb = pd.Timedelta(days=_embargo_days(horizon))
    tr = d < (pd.Timestamp("2025-07-01") - emb)
    va = (d >= pd.Timestamp("2025-07-01")) & (d < pd.Timestamp("2025-10-01"))
    te = (d >= pd.Timestamp("2025-10-01")) & (d < pd.Timestamp("2026-01-01"))
    # big-move outer target (news's best shot)
    k = max(abs(t) for t in thresholds)
    up, dn = y_all[_target_col(k)], y_all[_target_col(-k)]
    yt = ((up == 1) | (dn == 0)).astype(float)

    def sample():
        return dict(
            n_estimators=int(rng.choice([200, 300, 400, 600])),
            num_leaves=int(rng.integers(15, 64)),
            learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05])),
            min_child_samples=int(rng.choice([50, 100, 200, 400])),
            colsample_bytree=float(rng.uniform(0.4, 0.9)),
            subsample=float(rng.uniform(0.7, 1.0)),
            reg_lambda=float(rng.uniform(0, 12)),
            reg_alpha=float(rng.uniform(0, 2)),
            subsample_freq=1, max_bin=127, n_jobs=-1, verbose=-1, random_state=42,
        )

    best, best_val = None, np.inf
    for _ in range(n_configs):
        cfg = sample()
        p = _fit_predict(LGBMClassifier(**cfg), X_all.loc[tr], yt[tr], X_all.loc[va])
        v = brier_score_loss(yt[va], p)
        if v < best_val:
            best, best_val = cfg, v
    # final TEST: retuned-news vs production-config price-only and retuned price-only
    trva = tr | va
    p_news = _fit_predict(LGBMClassifier(**best), X_all.loc[trva], yt[trva], X_all.loc[te])
    p_price_prod = _fit_predict(_make_classifier("lightgbm"), X_all.loc[trva, price_cols],
                                yt[trva], X_all.loc[te, price_cols])
    p_price_retuned = _fit_predict(LGBMClassifier(**best), X_all.loc[trva, price_cols],
                                   yt[trva], X_all.loc[te, price_cols])
    return {
        "test_brier_retuned_news": brier_score_loss(yt[te], p_news),
        "test_brier_price_prod": brier_score_loss(yt[te], p_price_prod),
        "test_brier_price_retuned": brier_score_loss(yt[te], p_price_retuned),
        "test_auc_retuned_news": roc_auc_score(yt[te], p_news),
        "test_auc_price_prod": roc_auc_score(yt[te], p_price_prod),
    }


def main() -> None:
    for horizon in HORIZONS:
        print(f"\n{'=' * 78}\nHORIZON {horizon}\n{'=' * 78}")
        X_all, y_all, m_all, thresholds = build_pooled(horizon)
        print(f"pooled rows: {len(X_all):,}  news coverage: "
              f"{X_all[NEWS_HISTORY_COLS].notna().mean().mean():.0%}")
        for model in ("logistic", "lightgbm"):
            dbri, dauc, (tw, tt) = walk_forward(X_all, y_all, m_all, thresholds, model, horizon)
            print(f"\n  [{model}]  (mean Δ over folds; Δ<0 Brier / Δ>0 AUC = news helps)")
            print(f"  {'target':>12} | {'ΔBrier':>9} {'±sd':>7} | {'ΔAUC':>8} {'±sd':>7} | folds")
            for name in dbri:
                b, a = np.array(dbri[name]), np.array(dauc[name])
                print(f"  {name:>12} | {b.mean():+9.4f} {b.std():7.4f} | "
                      f"{np.nanmean(a):+8.3f} {np.nanstd(a):7.3f} | {len(b)}")
            allb = np.array([x for v in dbri.values() for x in v])
            alla = np.array([x for v in dauc.values() for x in v])
            print(f"  {'OVERALL':>12} | {allb.mean():+9.4f} {'':7} | {np.nanmean(alla):+8.3f} "
                  f"{'':7} | news-helps cells: Brier {np.mean(allb < 0):.0%}, "
                  f"AUC {np.mean(alla > 0):.0%}")
            if tt:
                print(f"  per-ticker win-rate (news lower Brier): {tw}/{tt} = {tw / tt:.0%}")

    print(f"\n{'=' * 78}\nLIGHTGBM RETUNE on 25 features (h20 big-move, held-out test)\n{'=' * 78}")
    X_all, y_all, m_all, thresholds = build_pooled(20)
    r = retune_lightgbm(X_all, y_all, m_all, thresholds)
    print(f"  retuned news   Brier {r['test_brier_retuned_news']:.4f}  AUC {r['test_auc_retuned_news']:.3f}")
    print(f"  price (prod)   Brier {r['test_brier_price_prod']:.4f}  AUC {r['test_auc_price_prod']:.3f}")
    print(f"  price (retuned)Brier {r['test_brier_price_retuned']:.4f}")
    win = r["test_brier_retuned_news"] < min(r["test_brier_price_prod"], r["test_brier_price_retuned"])
    print(f"  → retuned news {'BEATS' if win else 'does NOT beat'} price-only on held-out test")


if __name__ == "__main__":
    main()
