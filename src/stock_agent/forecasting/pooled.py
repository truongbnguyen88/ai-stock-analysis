"""Pooled (cross-sectional) ML model: artifact + training.

Trains ONE classifier per return-threshold on the stacked price-feature rows from
a universe of tickers, rather than one model per ticker. Per-ticker overlapping
windows give a low effective sample size; pooling across ~50-100 names yields
tens of thousands of rows and a model that generalizes (valid because the price
features are scale-free — ratios and bounded indicators).

The fitted classifiers + a fitted imputer + metadata are bundled into a
``PooledModel`` artifact, persisted with joblib, and loaded at inference. The
model is PRICE-ONLY by design (Option A): news sentiment is display context, not
a model input — see docs/TASKS.md decision log.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from stock_agent.features.assembler import _target_col, build_training_matrix
from stock_agent.features.price_features import PRICE_FEATURE_COLS
from stock_agent.forecasting.buckets import thresholds_for_horizon
from stock_agent.logging_config import get_logger
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)

ModelType = Literal["logistic", "xgboost", "lightgbm", "random_forest"]
_IMPUTED_MODELS = ("logistic", "random_forest")  # tree boosters handle NaN natively


def _make_classifier(model_type: ModelType) -> Any:
    """Construct an unfitted classifier of the requested type."""
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Standardize first: features span very different scales (RSI 0–100 vs
        # daily returns ~0.01), and unscaled logistic regression is ill-conditioned
        # and slow to converge. Trees are scale-invariant, so only the linear model
        # needs this. The upstream median imputer guarantees no NaNs reach the scaler.
        #
        # NOTE: NO class_weight="balanced". Backtesting (see validations_results.md)
        # showed balancing inflates rare-tail discrimination (AUC) but badly destroys
        # calibration (ECE up to 0.29) → far worse Brier. Plain logistic keeps the
        # tail signal AND stays well-calibrated, and beat every booster here.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000),
        )
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        # Regularized config (validated on big-move, see docs/validations_results.md):
        # the un-regularized default overfit → overconfident. With subsampling + L1/L2 +
        # min_child_samples it's the best big-move model for HIGH-VOL names (beats logistic
        # and the baseline on NVDA), while logistic stays best for stable names.
        return LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=300,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=8.0,
            reg_alpha=1.0,
            n_jobs=-1,
            verbose=-1,
        )
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", n_jobs=-1
        )
    raise ValueError(f"unknown model_type: {model_type}")


@dataclass
class PooledModel:
    """A trained pooled model: one classifier per threshold + metadata."""

    model_type: str
    horizon_days: int
    feature_cols: list[str]
    classifiers: dict[float, Any]
    imputer: Any | None  # fitted SimpleImputer for logistic/RF; None for boosters
    n_train_rows: int
    n_tickers: int
    trained_at: str
    notes: list[str] = field(default_factory=list)
    # Return cut-points this model was trained on (horizon-scaled). Empty on
    # artifacts trained before horizon-scaling -> derived from the horizon on use.
    thresholds: list[float] = field(default_factory=list)

    def effective_thresholds(self) -> list[float]:
        """The model's thresholds, deriving from horizon for pre-scaling artifacts.

        ``getattr`` (not ``self.thresholds``) so artifacts pickled before the field
        existed — which lack the attribute entirely — fall back to the horizon.
        """
        return getattr(self, "thresholds", None) or thresholds_for_horizon(self.horizon_days)

    def predict_exceedance(self, x_row: pd.DataFrame) -> list[float | None]:
        """P(return > threshold) for each threshold; None where a clf is absent."""
        X = x_row[self.feature_cols]
        if self.imputer is not None:
            X = pd.DataFrame(self.imputer.transform(X), columns=self.feature_cols, index=X.index)
        out: list[float | None] = []
        for thresh in self.effective_thresholds():
            clf = self.classifiers.get(thresh)
            out.append(float(clf.predict_proba(X)[0, 1]) if clf is not None else None)
        return out

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("pooled.saved", path=str(path), tickers=self.n_tickers, rows=self.n_train_rows)

    @staticmethod
    def load(path: Path) -> PooledModel:
        import joblib

        model: PooledModel = joblib.load(path)
        return model


def default_model_path(models_dir: Path, model_type: str, horizon_days: int) -> Path:
    """Canonical artifact path for a (model_type, horizon) pair."""
    return Path(models_dir) / f"pooled_{model_type}_h{horizon_days}.joblib"


def train_pooled_from_series(
    series_list: Sequence[PriceSeries],
    *,
    horizon_days: int,
    model_type: ModelType = "xgboost",
    min_rows_per_ticker: int = 60,
    min_total_rows: int = 500,
    earnings_by_ticker: dict[str, list[Date]] | None = None,
    vix: pd.Series | None = None,
) -> PooledModel:
    """Train a pooled model from in-memory price series (pure; offline-testable).

    Each ticker contributes its point-in-time (X, y) rows (built with leakage
    discipline by the assembler); rows are stacked, then one classifier per
    threshold is fit on the pool. Tickers with too little history are skipped.
    ``earnings_by_ticker`` supplies per-ticker earnings dates for the
    ``days_to_next_earnings`` feature (NaN where absent).
    """
    earnings_by_ticker = earnings_by_ticker or {}
    thresholds = thresholds_for_horizon(horizon_days)  # horizon-scaled cut-points
    frames_x: list[pd.DataFrame] = []
    frames_y: list[pd.DataFrame] = []
    used = 0
    for series in series_list:
        try:
            X, y = build_training_matrix(
                series,
                horizon_days,
                min_rows=min_rows_per_ticker,
                earnings_dates=earnings_by_ticker.get(series.ticker.upper()),
                vix=vix,
                thresholds=thresholds,
            )
        except ValueError as exc:
            log.debug("pooled.skip_ticker", ticker=series.ticker, reason=str(exc))
            continue
        frames_x.append(X)
        frames_y.append(y)
        used += 1

    if not frames_x:
        raise ValueError("no tickers produced usable training rows")

    X_all = pd.concat(frames_x, ignore_index=True)
    y_all = pd.concat(frames_y, ignore_index=True)
    # Ratio features (e.g. vol_ratio) can produce inf on degenerate windows;
    # treat as missing so the imputer / boosters handle them uniformly.
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    if len(X_all) < min_total_rows:
        raise ValueError(f"only {len(X_all)} pooled rows (need {min_total_rows})")

    needs_imputation = model_type in _IMPUTED_MODELS
    imputer = None
    X_fit = X_all
    if needs_imputation:
        from sklearn.impute import SimpleImputer

        # Fit the imputer on the POOLED training data and persist it, so inference
        # uses the same fill values (avoids the per-row imputation leak).
        # keep_empty_features=True: an all-NaN feature (e.g. days_to_next_earnings
        # when earnings data is unavailable) is filled with 0, not dropped — keeps
        # the column count stable across fit/transform.
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_all)
        X_fit = pd.DataFrame(
            imputer.transform(X_all), columns=PRICE_FEATURE_COLS, index=X_all.index
        )

    classifiers: dict[float, Any] = {}
    notes: list[str] = []
    for thresh in thresholds:
        labels = y_all[_target_col(thresh)]
        if labels.nunique() < 2:
            notes.append(f"threshold {thresh:+.2f} skipped (single class)")
            continue
        clf = _make_classifier(model_type)
        clf.fit(X_fit, labels)
        classifiers[thresh] = clf

    if not classifiers:
        raise ValueError("no threshold produced a trainable classifier")

    log.info(
        "pooled.trained",
        model=model_type,
        horizon=horizon_days,
        tickers=used,
        rows=len(X_all),
        thresholds=len(classifiers),
    )
    return PooledModel(
        model_type=model_type,
        horizon_days=horizon_days,
        feature_cols=list(PRICE_FEATURE_COLS),
        classifiers=classifiers,
        imputer=imputer,
        n_train_rows=len(X_all),
        n_tickers=used,
        trained_at=datetime.now(UTC).isoformat(),
        notes=notes,
        thresholds=list(thresholds),
    )
