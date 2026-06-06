"""Structural sanity check for trained pooled artifacts (network-free).

Used by the scheduled retrain (CI) as a promote gate: if any artifact is missing,
mis-shaped, or can't produce valid probabilities, the run fails and the previous
model release stays published. This catches *gross* failures — a crashed or
degenerate training run from a bad data month — cheaply, without an expensive
walk-forward backtest (which stays a manual/occasional check).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from stock_agent.features.price_features import PRICE_FEATURE_COLS
from stock_agent.forecasting.buckets import thresholds_for_horizon
from stock_agent.forecasting.conformal_calibrate import CONFORMAL_FILE, ConformalArtifact
from stock_agent.forecasting.pooled import PooledModel, default_model_path


def verify_artifacts(
    models_dir: Path,
    *,
    models: list[str],
    horizons: list[int],
    min_tickers: int = 0,
    min_rows: int = 0,
) -> list[str]:
    """Return a list of problems with the trained artifacts (empty == all OK).

    Per (model, horizon): the file exists and loads; its thresholds match the
    horizon's scaled cut-points; at least n-1 of n threshold-classifiers trained
    (one may be skipped as single-class); and a dummy-feature ``predict_exceedance``
    returns probabilities in [0, 1]. No network or market data required.

    ``min_tickers`` / ``min_rows`` add a **data-quality gate**: an artifact that
    trained on too small a slice of the universe is rejected even if structurally
    valid. This catches a degraded data month (e.g. yfinance rate-limited from the
    CI runner returns a handful of tickers) — a near-empty model still satisfies the
    structural checks, so without this it would silently publish. Defaults of 0
    disable the gate (used by structural-only unit tests).
    """
    problems: list[str] = []
    dummy = pd.DataFrame([{c: 0.0 for c in PRICE_FEATURE_COLS}])
    for model in models:
        for horizon in horizons:
            tag = f"{model} h{horizon}"
            path = default_model_path(models_dir, model, horizon)
            if not path.exists():
                problems.append(f"{tag}: artifact missing ({path.name})")
                continue
            try:
                m = PooledModel.load(path)
            except Exception as exc:  # noqa: BLE001 - any load failure is a problem
                problems.append(f"{tag}: failed to load ({exc})")
                continue

            expected = thresholds_for_horizon(horizon)
            if list(m.thresholds) != expected:
                problems.append(f"{tag}: thresholds {m.thresholds} != expected {expected}")
            if len(m.classifiers) < len(expected) - 1:
                problems.append(
                    f"{tag}: only {len(m.classifiers)}/{len(expected)} threshold-classifiers"
                )
            if m.n_tickers < min_tickers:
                problems.append(
                    f"{tag}: trained on {m.n_tickers} tickers (need >= {min_tickers}) "
                    "— degraded data month?"
                )
            if m.n_train_rows < min_rows:
                problems.append(
                    f"{tag}: trained on {m.n_train_rows:,} rows (need >= {min_rows:,}) "
                    "— degraded data month?"
                )
            try:
                probs = m.predict_exceedance(dummy)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{tag}: predict_exceedance failed ({exc})")
                continue
            defined = [p for p in probs if p is not None]
            if not defined:
                problems.append(f"{tag}: no threshold produced a probability")
            elif any(not (0.0 <= p <= 1.0) for p in defined):
                problems.append(f"{tag}: probability outside [0, 1]")
    return problems


def verify_conformal(
    models_dir: Path,
    *,
    required_models: list[str],
    horizons: list[int],
    ci_level: float = 0.90,
    min_coverage_after: float = 0.85,
) -> list[str]:
    """Return problems with the conformal interval-correction artifact (empty == OK).

    Promote-gate check that ``conformal.json`` exists, claims the expected ``ci_level``,
    and — for every required (model, horizon) — carries a finite ``q`` whose post-
    correction OOS coverage reached ~target (``>= min_coverage_after``). A missing
    artifact or a cell where conformal failed to achieve coverage (e.g. a degraded
    calibration month) fails the gate, so a release never ships mis-sized served CIs.
    """
    path = models_dir / CONFORMAL_FILE
    art = ConformalArtifact.load(path)
    if art is None:
        return [f"conformal: artifact missing ({CONFORMAL_FILE})"]
    problems: list[str] = []
    if abs(art.ci_level - ci_level) > 1e-6:
        problems.append(f"conformal: ci_level {art.ci_level} != expected {ci_level}")
    for model in required_models:
        for horizon in horizons:
            tag = f"conformal {model} h{horizon}"
            entry = art.entries.get(model, {}).get(horizon)
            if entry is None:
                problems.append(f"{tag}: missing")
                continue
            if not math.isfinite(entry.q):
                problems.append(f"{tag}: non-finite q")
            if not (min_coverage_after <= entry.coverage_after <= 1.0 + 1e-9):
                problems.append(
                    f"{tag}: post-correction coverage {entry.coverage_after:.2f} "
                    f"below {min_coverage_after:.2f}"
                )
    return problems
