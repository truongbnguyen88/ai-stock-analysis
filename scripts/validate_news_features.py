"""OOS A/B: does adding GDELT news features beat price-only? (Task 10 validation)

Reuses the REAL training building blocks so the comparison is faithful:
  - ``build_training_matrix`` → price features + horizon-scaled targets (leakage-safe)
  - ``build_news_history_features`` → the new scale-free, lagged news columns
  - ``_make_classifier("logistic")`` → the production scaled-logistic
  - SimpleImputer(median) → same imputation as the pooled trainer

Design:
  - Pool a basket of dense-news, liquid names (volatile semis + stable large-caps).
  - Restrict to rows where news exists (>= NEWS_START) so BOTH arms use identical
    rows — the only difference is the presence of the news columns.
  - TEMPORAL split (train < SPLIT, test >= SPLIT): no random folds, no leakage.
  - Per threshold, fit price-only vs price+news and compare OOS Brier / ROC-AUC.

Run:  PYTHONPATH=src python scripts/validate_news_features.py
"""

from __future__ import annotations

import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
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
from stock_agent.forecasting.train_pooled import (
    fetch_universe_earnings,
    fetch_universe_series,
)
from stock_agent.providers.registry import build_default_registry
from stock_agent.settings import get_settings

warnings.filterwarnings("ignore")

BASKET = [
    "NVDA", "AMD", "AAPL", "MSFT", "TSLA", "AMZN", "META", "GOOGL",
    "AVGO", "MU", "GS", "JPM", "WMT", "KO", "BA",
]
HORIZON = 20
NEWS_START = pd.Timestamp("2023-04-01")  # after buzz-window warmup
SPLIT = pd.Timestamp("2025-09-01")  # train < SPLIT <= test
MODEL = os.environ.get("AB_MODEL", "logistic")  # logistic | lightgbm


def _pooled_rows(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build pooled (X_price+news, y, dates) across the basket, news-covered rows only."""
    registry = build_default_registry(get_settings())
    series = fetch_universe_series(BASKET, registry)
    earnings = fetch_universe_earnings([s.ticker for s in series], registry)
    store = load_news_store()
    thresholds = thresholds_for_horizon(horizon)

    vix_end = date.today()
    vix = fetch_vix(registry, start=date(2018, 1, 1), end=vix_end)

    xs, ys, dates = [], [], []
    for s in series:
        try:
            X, y = build_training_matrix(
                s, horizon, earnings_dates=earnings.get(s.ticker.upper()),
                vix=vix, thresholds=thresholds,
            )
        except ValueError:
            continue
        news = build_news_history_features(s.ticker, pd.DatetimeIndex(X.index), store)
        Xn = pd.concat([X, news], axis=1)
        mask = pd.DatetimeIndex(Xn.index) >= NEWS_START
        if mask.sum() == 0:
            continue
        xs.append(Xn.loc[mask])
        ys.append(y.loc[mask])
        dates.append(pd.Series(pd.DatetimeIndex(Xn.index)[mask]))
        print(f"  {s.ticker:5s} {int(mask.sum()):5d} rows")

    X_all = pd.concat(xs, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    y_all = pd.concat(ys, ignore_index=True)
    d_all = pd.concat(dates, ignore_index=True)
    return X_all, y_all, d_all


def _fit_eval(Xtr, ytr, Xte, yte) -> tuple[float, float]:
    """Fit scaled-logistic on (Xtr,ytr); return (Brier, AUC) on the test fold."""
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    Xtr_i = imp.transform(Xtr)
    Xte_i = imp.transform(Xte)
    clf = _make_classifier(MODEL)
    clf.fit(Xtr_i, ytr)
    p = clf.predict_proba(Xte_i)[:, 1]
    brier = brier_score_loss(yte, p)
    auc = roc_auc_score(yte, p) if yte.nunique() > 1 else float("nan")
    return brier, auc


def main() -> None:
    print(f"Building pooled rows (h={HORIZON}, news from {NEWS_START.date()}) …")
    X_all, y_all, d_all = _pooled_rows(HORIZON)
    price_cols = [c for c in X_all.columns if c not in NEWS_HISTORY_COLS]

    tr = d_all < SPLIT
    te = d_all >= SPLIT
    print(f"\nMODEL={MODEL}  Pooled rows: {len(X_all):,}  |  "
          f"train {int(tr.sum()):,}  test {int(te.sum()):,}")
    print(f"News coverage in rows: {X_all[NEWS_HISTORY_COLS].notna().mean().mean():.1%}\n")

    thresholds = thresholds_for_horizon(HORIZON)
    print(f"{'threshold':>10} | {'Brier price':>11} {'Brier +news':>11} {'ΔBrier':>8} "
          f"| {'AUC price':>9} {'AUC +news':>9} {'ΔAUC':>7}")
    print("-" * 82)
    agg = {"bp": [], "bn": [], "ap": [], "an": []}
    for thr in thresholds:
        col = _target_col(thr)
        ytr, yte = y_all.loc[tr, col], y_all.loc[te, col]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            print(f"{thr:+10.2f} | (single-class fold — skipped)")
            continue
        bp, ap = _fit_eval(X_all.loc[tr, price_cols], ytr, X_all.loc[te, price_cols], yte)
        bn, an = _fit_eval(X_all.loc[tr], ytr, X_all.loc[te], yte)
        agg["bp"].append(bp); agg["bn"].append(bn); agg["ap"].append(ap); agg["an"].append(an)
        print(f"{thr:+10.2f} | {bp:11.4f} {bn:11.4f} {bn - bp:+8.4f} "
              f"| {ap:9.3f} {an:9.3f} {an - ap:+7.3f}")
    print("-" * 82)
    print(f"{'MEAN':>10} | {np.mean(agg['bp']):11.4f} {np.mean(agg['bn']):11.4f} "
          f"{np.mean(agg['bn']) - np.mean(agg['bp']):+8.4f} "
          f"| {np.mean(agg['ap']):9.3f} {np.mean(agg['an']):9.3f} "
          f"{np.mean(agg['an']) - np.mean(agg['ap']):+7.3f}")

    # Big-move target |r| > k = (r > +k) OR (r < -k) = (y[+k]==1) | (y[-k]==0).
    # This is ML's only historical edge (volatility), so it's news's best shot.
    print(f"\n{'big |r|>k':>10} | {'Brier price':>11} {'Brier +news':>11} {'ΔBrier':>8} "
          f"| {'AUC price':>9} {'AUC +news':>9} {'ΔAUC':>7}")
    print("-" * 82)
    for k in (0.05, 0.10):
        up, dn = y_all[_target_col(k)], y_all[_target_col(-k)]
        big = ((up == 1) | (dn == 0)).astype(float)
        ytr, yte = big.loc[tr], big.loc[te]
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue
        bp, ap = _fit_eval(X_all.loc[tr, price_cols], ytr, X_all.loc[te, price_cols], yte)
        bn, an = _fit_eval(X_all.loc[tr], ytr, X_all.loc[te], yte)
        print(f"{k:+10.2f} | {bp:11.4f} {bn:11.4f} {bn - bp:+8.4f} "
              f"| {ap:9.3f} {an:9.3f} {an - ap:+7.3f}")

    print("\nΔ negative Brier = news helps; Δ positive AUC = news helps.")


if __name__ == "__main__":
    main()
