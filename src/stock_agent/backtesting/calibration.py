"""Probability calibration: reliability diagrams, ECE, and post-hoc fits.

Calibration asks: of all the times the model said "70%", did the event happen
~70% of the time? This is the project's headline *trustworthiness* measure — a
model can rank well (good AUC) yet be badly calibrated, which would make the
reported probabilities misleading.

- ``reliability_bins`` / ``expected_calibration_error``: equal-width binning of
  predicted probability; ECE is the bin-count-weighted |confidence − accuracy|.
- ``fit_isotonic`` / ``fit_platt``: post-hoc recalibration maps (monotone
  non-parametric, and logistic-on-the-logit respectively).
- ``calibration_report``: pools all OOS predictions, reports raw ECE/MCE, and —
  to answer "would recalibration help?" honestly — fits a recalibration map on
  an EARLIER half and evaluates ECE on a held-out LATER half (no fit/eval leak).

All pure given the inputs; the sklearn fits are deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from stock_agent.schemas.backtest import CalibrationReport, ReliabilityBin

Calibrator = Callable[[np.ndarray], np.ndarray]


def _arrays(
    p: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pa = np.asarray(p, dtype=float)
    ya = np.asarray(y, dtype=float)
    if pa.shape != ya.shape:
        raise ValueError(f"p and y shape mismatch: {pa.shape} vs {ya.shape}")
    if pa.size == 0:
        raise ValueError("empty calibration input")
    return pa, ya


def reliability_bins(
    p: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, *, n_bins: int = 10
) -> list[ReliabilityBin]:
    """Bin predictions into ``n_bins`` equal-width [0,1] bins (non-empty only).

    For each bin: ``mean_pred`` = mean predicted probability (confidence),
    ``frequency`` = realized positive rate (accuracy). The top bin is closed on
    the right so ``p == 1.0`` lands in it.
    """
    pa, ya = _arrays(p, y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False gives bins 1..n_bins for p in (0,1]; clip p==0
    # into the first bin and p==1 into the last.
    idx = np.clip(np.digitize(pa, edges[1:-1], right=False), 0, n_bins - 1)
    out: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        out.append(
            ReliabilityBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                count=count,
                mean_pred=float(pa[mask].mean()),
                frequency=float(ya[mask].mean()),
            )
        )
    return out


def expected_calibration_error(
    p: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray, *, n_bins: int = 10
) -> tuple[float, float]:
    """Return ``(ECE, MCE)``.

    ECE = Σ_b (n_b / N) · |conf_b − acc_b|  (bin-weighted average gap).
    MCE = max_b |conf_b − acc_b|            (worst-case gap).
    """
    pa, ya = _arrays(p, y)
    bins = reliability_bins(pa, ya, n_bins=n_bins)
    n = pa.size
    ece = sum((b.count / n) * abs(b.mean_pred - b.frequency) for b in bins)
    mce = max((abs(b.mean_pred - b.frequency) for b in bins), default=0.0)
    return float(ece), float(mce)


def calibration_label(ece: float) -> str:
    """Plain-language trust label from ECE (the avg |confidence − accuracy| gap).

    Conventional rule-of-thumb cutoffs, not hard guarantees — surfaced to the
    agent/user so "is this forecast trustworthy?" gets an honest, bucketed answer
    alongside the raw ECE number. Lower ECE = better calibrated.
    """
    if ece < 0.05:
        return "well-calibrated"
    if ece < 0.10:
        return "moderately calibrated"
    return "poorly calibrated"


def fit_isotonic(p: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> Calibrator:
    """Fit a monotone (isotonic) recalibration map ``p -> P(y=1)``.

    Non-parametric and flexible; clips out-of-range inputs at apply time.
    """
    from sklearn.isotonic import IsotonicRegression

    pa, ya = _arrays(p, y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(pa, ya)

    def apply(x: np.ndarray) -> np.ndarray:
        return np.asarray(iso.predict(np.asarray(x, dtype=float)), dtype=float)

    return apply


def fit_platt(p: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> Calibrator:
    """Fit Platt scaling: logistic regression on the logit of the raw probability.

    Parametric (a sigmoid), robust with little data; assumes a sigmoidal
    distortion. Inputs are clipped before the logit to avoid ±inf.
    """
    from sklearn.linear_model import LogisticRegression

    pa, ya = _arrays(p, y)
    z = _logit(pa).reshape(-1, 1)
    # If labels are single-class, Platt cannot fit — fall back to identity.
    if ya.min() == ya.max():
        return lambda x: np.asarray(x, dtype=float)
    lr = LogisticRegression().fit(z, ya)

    def apply(x: np.ndarray) -> np.ndarray:
        zx = _logit(np.asarray(x, dtype=float)).reshape(-1, 1)
        return np.asarray(lr.predict_proba(zx)[:, 1], dtype=float)

    return apply


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    pc = np.clip(p, eps, 1.0 - eps)
    return np.asarray(np.log(pc / (1.0 - pc)), dtype=float)


def calibration_report(
    p: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
    post_method: str | None = "isotonic",
    min_post_samples: int = 100,
) -> CalibrationReport:
    """Build the full calibration report over pooled, time-ordered predictions.

    ``p``/``y`` MUST be in chronological order so the post-hoc split is a true
    earlier-fit / later-evaluate holdout (no leakage). If there are fewer than
    ``2 * min_post_samples`` points, the post-hoc fields are left ``None``.
    """
    pa, ya = _arrays(p, y)
    ece, mce = expected_calibration_error(pa, ya, n_bins=n_bins)
    bins = reliability_bins(pa, ya, n_bins=n_bins)

    ece_pre: float | None = None
    ece_post: float | None = None
    method_used: str | None = None
    if post_method is not None and pa.size >= 2 * min_post_samples:
        # Temporal holdout: fit recalibration on the earlier half, evaluate later.
        cut = pa.size // 2
        p_fit, y_fit = pa[:cut], ya[:cut]
        p_eval, y_eval = pa[cut:], ya[cut:]
        calibrator = (
            fit_isotonic(p_fit, y_fit) if post_method == "isotonic" else fit_platt(p_fit, y_fit)
        )
        ece_pre, _ = expected_calibration_error(p_eval, y_eval, n_bins=n_bins)
        ece_post, _ = expected_calibration_error(calibrator(p_eval), y_eval, n_bins=n_bins)
        method_used = post_method

    return CalibrationReport(
        n=int(pa.size),
        n_bins=n_bins,
        bins=bins,
        ece=ece,
        mce=mce,
        method_post=method_used,
        ece_pre_holdout=ece_pre,
        ece_post=ece_post,
    )
