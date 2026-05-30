"""Binary-classification metrics for probabilistic forecasts.

Each forecaster emits, at every as-of, an exceedance probability ``P(r > θ)``
for each return threshold θ; the realized label is ``1[r > θ]``. These pure
functions score the pooled out-of-sample (probability, label) pairs.

The simple metrics are hand-rolled (small, golden-testable, explicit clipping);
ROC AUC delegates to scikit-learn (already a dependency) and returns ``None``
when only one class is present (AUC undefined).

Convention: ``y`` are 0/1 labels, ``p`` are probabilities in [0, 1]; a 0.5
decision threshold is used for accuracy/precision/recall.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stock_agent.schemas.backtest import ThresholdMetrics

_EPS = 1e-15  # log-loss clip so log(0) never occurs


def _arrays(
    y: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ya = np.asarray(y, dtype=float)
    pa = np.asarray(p, dtype=float)
    if ya.shape != pa.shape:
        raise ValueError(f"y and p shape mismatch: {ya.shape} vs {pa.shape}")
    if ya.size == 0:
        raise ValueError("empty metric input")
    return ya, pa


def base_rate(y: Sequence[float] | np.ndarray) -> float:
    """Empirical positive rate ``P(label = 1)``."""
    ya = np.asarray(y, dtype=float)
    return float(ya.mean()) if ya.size else 0.0


def brier_score(y: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray) -> float:
    """Mean squared error between probability and outcome (lower is better)."""
    ya, pa = _arrays(y, p)
    return float(np.mean((pa - ya) ** 2))


def log_loss(y: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray) -> float:
    """Binary cross-entropy with probabilities clipped to [eps, 1-eps]."""
    ya, pa = _arrays(y, p)
    pc = np.clip(pa, _EPS, 1.0 - _EPS)
    return float(-np.mean(ya * np.log(pc) + (1.0 - ya) * np.log(1.0 - pc)))


def accuracy(
    y: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray, *, threshold: float = 0.5
) -> float:
    """Fraction correct when predicting positive iff ``p >= threshold``."""
    ya, pa = _arrays(y, p)
    pred = (pa >= threshold).astype(float)
    return float(np.mean(pred == ya))


def precision(
    y: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray, *, threshold: float = 0.5
) -> float | None:
    """TP / (TP + FP); ``None`` when nothing is predicted positive."""
    ya, pa = _arrays(y, p)
    pred = pa >= threshold
    tp = float(np.sum((pred) & (ya == 1)))
    fp = float(np.sum((pred) & (ya == 0)))
    return tp / (tp + fp) if (tp + fp) > 0 else None


def recall(
    y: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray, *, threshold: float = 0.5
) -> float | None:
    """TP / (TP + FN); ``None`` when there are no actual positives."""
    ya, pa = _arrays(y, p)
    pred = pa >= threshold
    tp = float(np.sum((pred) & (ya == 1)))
    fn = float(np.sum((~pred) & (ya == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else None


def roc_auc(y: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray) -> float | None:
    """Area under the ROC curve; ``None`` if only one class is present.

    Delegates to scikit-learn (handles ties exactly). AUC is the probability a
    random positive is ranked above a random negative — a threshold-free measure
    of discrimination.
    """
    ya, pa = _arrays(y, p)
    if ya.min() == ya.max():  # single class → AUC undefined
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(ya, pa))


def threshold_metrics(
    y: Sequence[float] | np.ndarray, p: Sequence[float] | np.ndarray, *, threshold: float
) -> ThresholdMetrics:
    """Bundle all metrics for one return threshold into a typed record."""
    ya, _ = _arrays(y, p)
    return ThresholdMetrics(
        threshold=threshold,
        n=int(ya.size),
        n_positive=int(ya.sum()),
        base_rate=base_rate(y),
        brier=brier_score(y, p),
        log_loss=log_loss(y, p),
        accuracy=accuracy(y, p),
        precision=precision(y, p),
        recall=recall(y, p),
        roc_auc=roc_auc(y, p),
    )
