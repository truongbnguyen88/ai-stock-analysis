"""Heavy-tune XGBoost + rigorous walk-forward A/B vs logistic & lightgbm, with news.

User request (post-Task-10): give XGBoost a second chance with the same tuning rigor
lightgbm got (config #38), then compare all three ML models — run WITH the news
features (currently the 7 base per-ticker + market features; topic features are NaN
until topics.csv is pulled in July).

Method:
  1. build_pooled: price + news features (log1p buzz), pooled over a 30-ticker basket.
  2. tune_xgboost: random search at h20 on a 20-ticker TUNE basket (train→val Brier on
     the big-move targets), then confirm the winner on a DISJOINT 10-ticker HELD-OUT
     basket (winner's-curse guard — does it generalize to unseen tickers?).
  3. compare: walk-forward folds at h20/30/60; logistic vs lightgbm vs tuned-xgboost
     (all with news) + a base-rate floor; per-target Brier/AUC averaged over folds.

Run: PYTHONPATH=src python scripts/tune_validate_xgboost.py
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from stock_agent.backtesting.calibration import calibration_label, expected_calibration_error
from stock_agent.data.market_context import fetch_vix
from stock_agent.features.assembler import _target_col, build_training_matrix
from stock_agent.features.news_history import build_news_history_features, load_news_store
from stock_agent.forecasting.buckets import thresholds_for_horizon
from stock_agent.forecasting.pooled import _make_classifier
from stock_agent.forecasting.train_pooled import fetch_universe_earnings, fetch_universe_series
from stock_agent.providers.registry import build_default_registry
from stock_agent.settings import get_settings

warnings.filterwarnings("ignore")

BASKET = [
    "NVDA", "AMD", "AVGO", "MU", "MRVL", "SMCI", "TSLA", "ARM", "PLTR", "MPWR",
    "AAPL", "MSFT", "AMZN", "META", "GOOGL", "NFLX", "ORCL", "CRM",
    "JPM", "GS", "BAC", "UNH", "LLY", "JNJ", "CAT", "BA", "WMT", "KO", "XOM", "CVX",
]
TUNE_TICKERS = set(BASKET[:20])          # tune here …
HELDOUT_TICKERS = set(BASKET[20:])       # … confirm the winner here (disjoint)
HORIZONS = [20, 30, 60]
TUNE_HORIZON = 20
N_CONFIGS = 60
NEWS_START = pd.Timestamp("2023-04-01")
BUZZ_COLS = ["news_buzz", "epu_buzz"]
# Only the 7 ACTIVE news features (per-ticker + market). The 10 topic columns are
# all-NaN until topics.csv is pulled, so they are excluded entirely here → a clean
# 18 price + 7 news = 25-feature matrix.
BASE_NEWS_COLS = [
    "news_buzz", "news_tone", "news_pos_frac", "news_neg_frac",
    "pol_tone", "epu_buzz", "pres_tone",
]
CCCV_CV = 5  # this run: cv=5 (production serves cv=3)
VAL_START = pd.Timestamp("2025-01-01")   # tuning train/val boundary
VAL_END = pd.Timestamp("2025-07-01")
FOLD_STARTS = [pd.Timestamp(s) for s in (
    "2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01",
    "2025-07-01", "2025-10-01", "2026-01-01",
)]
FOLD_LEN = pd.DateOffset(months=3)
NEEDS_IMPUTE = {"logistic"}


def build_pooled(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[float]]:
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
        news = build_news_history_features(s.ticker, pd.DatetimeIndex(X.index), store)[
            BASE_NEWS_COLS
        ].copy()
        news[BUZZ_COLS] = np.log1p(news[BUZZ_COLS])
        Xn = pd.concat([X, news], axis=1)
        dates = pd.DatetimeIndex(Xn.index)
        mask = dates >= NEWS_START
        if mask.sum() == 0:
            continue
        xs.append(Xn.loc[mask])
        ys.append(y.loc[mask])
        meta.append(pd.DataFrame({"date": dates[mask], "ticker": s.ticker}))
    X_all = pd.concat(xs, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return X_all, pd.concat(ys, ignore_index=True), pd.concat(meta, ignore_index=True), thresholds


def targets(y_all: pd.DataFrame, thresholds: list[float]):
    for thr in thresholds:
        yield f"dir {thr:+.2f}", y_all[_target_col(thr)]
    inner = min(abs(t) for t in thresholds if t != 0)
    outer = max(abs(t) for t in thresholds)
    for k in (inner, outer):
        up, dn = y_all[_target_col(k)], y_all[_target_col(-k)]
        yield f"big|r|>{k:.2f}", ((up == 1) | (dn == 0)).astype(float)


def big_move_targets(y_all, thresholds):
    inner = min(abs(t) for t in thresholds if t != 0)
    outer = max(abs(t) for t in thresholds)
    out = {}
    for k in (inner, outer):
        up, dn = y_all[_target_col(k)], y_all[_target_col(-k)]
        out[f"big{k}"] = ((up == 1) | (dn == 0)).astype(float)
    return out


def make_model(name: str, xgb_cfg: dict | None = None):
    if name == "xgboost":
        return XGBClassifier(**xgb_cfg)
    return _make_classifier(name)


def fit_predict(name, clf, Xtr, ytr, Xte, *, calibrate=False):
    """Fit and return P(class=1) on Xte.

    ``calibrate`` wraps the estimator in CalibratedClassifierCV(cv=3, isotonic) —
    the SAME post-hoc calibration the production pooled models use — so the model
    comparison is on production footing. Falls back to a raw fit if a fold has too
    few minority samples for cv=3 (CCCV clones the estimator, so clf stays unfitted).
    """
    if name in NEEDS_IMPUTE:
        imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
        Xtr, Xte = imp.transform(Xtr), imp.transform(Xte)
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV

        try:
            cal = CalibratedClassifierCV(clf, cv=CCCV_CV, method="isotonic").fit(Xtr, ytr)
            return cal.predict_proba(Xte)[:, 1]
        except ValueError:
            pass  # too few minority samples for cv=k → raw fit below
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def sample_xgb(rng) -> dict:
    return dict(
        n_estimators=int(rng.choice([200, 300, 400, 600])),
        max_depth=int(rng.choice([3, 4, 5, 6, 8])),
        learning_rate=float(rng.choice([0.01, 0.02, 0.03, 0.05])),
        min_child_weight=float(rng.choice([1, 5, 20, 50, 100])),
        subsample=float(rng.uniform(0.6, 1.0)),
        colsample_bytree=float(rng.uniform(0.4, 0.9)),
        reg_lambda=float(rng.uniform(0, 12)),
        reg_alpha=float(rng.uniform(0, 2)),
        gamma=float(rng.uniform(0, 1.0)),
        tree_method="hist", n_jobs=-1, eval_metric="logloss",
        random_state=42, verbosity=0,
    )


def _brier_on(cfg_or_name, X, y_targets, tr, te, xgb_cfg=None):
    """Mean Brier across the given target dict for one model on one split."""
    briers = []
    for yt in y_targets.values():
        if yt[tr].nunique() < 2 or yt[te].nunique() < 2:
            continue
        clf = make_model(cfg_or_name, xgb_cfg)
        p = fit_predict(cfg_or_name, clf, X.loc[tr], yt[tr], X.loc[te])
        briers.append(brier_score_loss(yt[te], p))
    return float(np.mean(briers)) if briers else np.nan


def tune_xgboost(X, y, meta, thresholds, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DatetimeIndex(meta["date"])
    tk = meta["ticker"].to_numpy()
    in_tune = np.isin(tk, list(TUNE_TICKERS))
    in_held = np.isin(tk, list(HELDOUT_TICKERS))
    before_val = np.asarray(d < VAL_START)
    in_val = np.asarray((d >= VAL_START) & (d < VAL_END))
    tr = in_tune & before_val
    va = in_tune & in_val
    bm = big_move_targets(y, thresholds)

    best_cfg, best_val = None, np.inf
    print(f"  tuning {N_CONFIGS} configs on {len(TUNE_TICKERS)} tickers "
          f"(train {tr.sum():,} / val {va.sum():,}) …")
    for i in range(N_CONFIGS):
        cfg = sample_xgb(rng)
        score = _brier_on("xgboost", X, bm, tr, va, xgb_cfg=cfg)
        if score < best_val:
            best_cfg, best_val = cfg, score
            print(f"    [{i:02d}] new best val Brier {score:.4f}  (depth={cfg['max_depth']} "
                  f"n={cfg['n_estimators']} lr={cfg['learning_rate']} λ={cfg['reg_lambda']:.1f})")

    # Winner's-curse guard: confirm on the DISJOINT held-out basket (same val period).
    htr = in_held & before_val
    hva = in_held & in_val
    print(f"\n  HELD-OUT confirmation ({len(HELDOUT_TICKERS)} unseen tickers, "
          f"train {htr.sum():,} / val {hva.sum():,}) — mean big-move Brier:")
    for name, cfg in (("xgboost(tuned)", best_cfg), ("logistic", None), ("lightgbm", None)):
        b = _brier_on("xgboost" if cfg else name, X, bm, htr, hva, xgb_cfg=cfg)
        print(f"    {name:>15}: {b:.4f}")
    return best_cfg


def walk_forward_compare(X, y, meta, thresholds, horizon, xgb_cfg):
    """Per-fold metrics → mean ± std across folds (the dispersion) + per-target Brier.

    Each fold's score for a model is the mean over targets of Brier / AUC / ECE, so
    a "fold" is one independent OOS sample. ECE is the bin-weighted calibration gap.
    """
    d = pd.DatetimeIndex(meta["date"])
    emb = pd.Timedelta(days=int(np.ceil(horizon * 1.5)))
    models = ["base_rate", "logistic", "lightgbm", "xgboost"]
    fold_scores = {m: {"brier": [], "auc": [], "ece": []} for m in models}  # per-fold means
    per_target = {}  # name -> {model -> [brier across folds]}

    for f0 in FOLD_STARTS:
        f1 = f0 + FOLD_LEN
        te = np.asarray((d >= f0) & (d < f1))
        tr = np.asarray(d < (f0 - emb))
        if te.sum() < 50 or tr.sum() < 500:
            continue
        fold = {m: {"brier": [], "auc": [], "ece": []} for m in models}
        for name, yt in targets(y, thresholds):
            ytr, yte = yt[tr], yt[te]
            if ytr.nunique() < 2 or yte.nunique() < 2:
                continue
            pt = per_target.setdefault(name, {m: [] for m in models})
            p0 = np.full(int(te.sum()), float(ytr.mean()))  # no-skill floor
            # All three ML models CCCV-calibrated (cv=3, isotonic) — production footing.
            preds = {
                "base_rate": p0,
                "logistic": fit_predict("logistic", make_model("logistic"), X.loc[tr], ytr, X.loc[te], calibrate=True),  # noqa: E501
                "lightgbm": fit_predict("lightgbm", make_model("lightgbm"), X.loc[tr], ytr, X.loc[te], calibrate=True),  # noqa: E501
                "xgboost": fit_predict("xgboost", make_model("xgboost", xgb_cfg), X.loc[tr], ytr, X.loc[te], calibrate=True),  # noqa: E501
            }
            for m, p in preds.items():
                fold[m]["brier"].append(brier_score_loss(yte, p))
                fold[m]["auc"].append(roc_auc_score(yte, p) if yte.nunique() > 1 else np.nan)
                fold[m]["ece"].append(expected_calibration_error(p, yte.to_numpy())[0])
                pt[m].append(brier_score_loss(yte, p))
        # collapse this fold's targets to one number per model
        for m in models:
            if fold[m]["brier"]:
                fold_scores[m]["brier"].append(np.mean(fold[m]["brier"]))
                fold_scores[m]["auc"].append(np.nanmean(fold[m]["auc"]))
                fold_scores[m]["ece"].append(np.mean(fold[m]["ece"]))

    n_folds = len(fold_scores["xgboost"]["brier"])
    print(f"\n  h{horizon}: mean ± std over {n_folds} folds — 18 price + 7 news, CCCV cv={CCCV_CV}")
    print(f"  {'model':>10} | {'Brier':>16} | {'AUC':>14} | {'ECE':>16} | trust")
    for m in models:
        b, a, e = (np.array(fold_scores[m][k]) for k in ("brier", "auc", "ece"))
        print(f"  {m:>10} | {b.mean():.4f} ± {b.std():.4f} | {np.nanmean(a):.3f} ± {np.nanstd(a):.3f}"
              f" | {e.mean():.4f} ± {e.std():.4f} | {calibration_label(float(e.mean()))}")
    print("  per-target Brier (best model in []):")
    for name, md in per_target.items():
        means = {m: float(np.mean(v)) for m, v in md.items()}
        best = min(means, key=means.get)
        cells = "  ".join(f"{m[:4]} {means[m]:.4f}" for m in models)
        print(f"    {name:>10} | {cells}   [{best}]")


def main() -> None:
    print(f"{'=' * 80}\nXGBoost heavy-tune + 3-way ML comparison (with news features)\n{'=' * 80}")
    Xt, yt, mt, thr = build_pooled(TUNE_HORIZON)
    print(f"pooled h{TUNE_HORIZON}: {len(Xt):,} rows, {Xt.shape[1]} features "
          f"({len(BASKET)} tickers; news cols: {BASE_NEWS_COLS})\n")
    best_cfg = tune_xgboost(Xt, yt, mt, thr)
    print(f"\nBEST XGB CONFIG: {best_cfg}\n")

    for h in HORIZONS:
        X, y, m, t = (Xt, yt, mt, thr) if h == TUNE_HORIZON else build_pooled(h)
        walk_forward_compare(X, y, m, t, h, best_cfg)


if __name__ == "__main__":
    main()
