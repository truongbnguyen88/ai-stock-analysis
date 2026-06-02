"""Structural sanity check for trained pooled artifacts (network-free).

Used by the scheduled retrain (CI) as a promote gate: if any artifact is missing,
mis-shaped, or can't produce valid probabilities, the run fails and the previous
model release stays published. This catches *gross* failures — a crashed or
degenerate training run from a bad data month — cheaply, without an expensive
walk-forward backtest (which stays a manual/occasional check).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_agent.features.price_features import PRICE_FEATURE_COLS
from stock_agent.forecasting.buckets import thresholds_for_horizon
from stock_agent.forecasting.pooled import PooledModel, default_model_path


def verify_artifacts(models_dir: Path, *, models: list[str], horizons: list[int]) -> list[str]:
    """Return a list of problems with the trained artifacts (empty == all OK).

    Per (model, horizon): the file exists and loads; its thresholds match the
    horizon's scaled cut-points; at least n-1 of n threshold-classifiers trained
    (one may be skipped as single-class); and a dummy-feature ``predict_exceedance``
    returns probabilities in [0, 1]. No network or market data required.
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
