"""Feature-group ablation: compare walk-forward metrics baseline vs baseline+group.

Pure aggregation/reporting over ``BacktestResult`` objects — no network, no model
training — so it is unit-testable. The networked driver that actually runs the
per-fold pooled refits lives in ``scripts/ablate_feature_groups.py``; it produces
the ``BacktestResult`` lists this module summarizes.

Decision rule (kept deliberately conservative): a group is a *promotion candidate*
only if it improves the headline out-of-sample metrics without degrading
calibration — i.e. lower pooled Brier and lower (or ~equal) ECE, and ideally a
better big-move score. We never promote on in-sample fit or on a single ticker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from stock_agent.schemas.backtest import BacktestResult

# Metrics where a LOWER value is better (used to sign the "improvement" deltas).
LOWER_IS_BETTER: frozenset[str] = frozenset({"mean_brier", "ece", "big_move_brier"})
# Metrics where a HIGHER value is better.
HIGHER_IS_BETTER: frozenset[str] = frozenset({"big_move_auc"})


def extract_metrics(result: BacktestResult) -> dict[str, float]:
    """Pull the headline OOS metrics from one backtest result.

    ``coverage_abs_gap`` is |empirical_coverage − ci_level| (interval honesty);
    missing optional metrics (no big-move / no conformal split) are omitted so the
    aggregator can skip them rather than treat absence as 0.
    """
    metrics: dict[str, float] = {
        "mean_brier": result.mean_brier,
        "ece": result.calibration.ece,
    }
    if result.big_move is not None:
        metrics["big_move_brier"] = result.big_move.brier
        if result.big_move.roc_auc is not None:
            metrics["big_move_auc"] = result.big_move.roc_auc
    if result.conformal is not None:
        metrics["coverage_abs_gap"] = abs(
            result.conformal.empirical_coverage - result.conformal.ci_level
        )
    return metrics


def aggregate(results: Sequence[BacktestResult]) -> dict[str, float]:
    """Mean of each metric across results (per-ticker), skipping absent values."""
    if not results:
        return {}
    rows = [extract_metrics(r) for r in results]
    keys = {k for row in rows for k in row}
    out: dict[str, float] = {}
    for k in keys:
        vals = [row[k] for row in rows if k in row]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def improvement(metric: str, baseline: float, candidate: float) -> float:
    """Signed improvement of candidate over baseline (positive = better).

    Normalizes metric direction so a positive number always means "the group
    helped". For lower-is-better metrics that's ``baseline - candidate``; for
    higher-is-better it's ``candidate - baseline``.
    """
    if metric in LOWER_IS_BETTER:
        return baseline - candidate
    if metric in HIGHER_IS_BETTER:
        return candidate - baseline
    # Direction-neutral (e.g. coverage gap is already an absolute distance, lower
    # better): treat as lower-is-better.
    return baseline - candidate


def ablation_table(
    by_label: Mapping[str, Sequence[BacktestResult]], *, baseline_label: str = "baseline"
) -> list[dict[str, object]]:
    """Build one row per label: aggregated metrics + signed improvement vs baseline.

    Raises if the baseline label is missing (the comparison is meaningless without it).
    """
    if baseline_label not in by_label:
        raise ValueError(f"baseline label {baseline_label!r} not in results")
    base = aggregate(by_label[baseline_label])
    rows: list[dict[str, object]] = []
    for label, results in by_label.items():
        agg = aggregate(results)
        row: dict[str, object] = {"label": label, "n_tickers": len(results)}
        for metric, value in agg.items():
            row[metric] = value
            if label != baseline_label and metric in base:
                row[f"d_{metric}"] = improvement(metric, base[metric], value)
        rows.append(row)
    return rows


def is_promotion_candidate(row: dict[str, object], *, min_brier_gain: float = 1e-4) -> bool:
    """Conservative promote gate for one ablation row (a candidate group).

    True iff Brier improves by at least ``min_brier_gain`` AND calibration (ECE)
    does not get worse. ``min_brier_gain`` filters noise-level deltas. The baseline
    row (no deltas) is never a candidate.
    """
    d_brier = row.get("d_mean_brier")
    d_ece = row.get("d_ece")
    if not isinstance(d_brier, (int, float)) or not isinstance(d_ece, (int, float)):
        return False
    return d_brier >= min_brier_gain and d_ece >= 0.0  # ECE improvement is positive


def render_markdown(rows: list[dict[str, object]]) -> str:
    """Render the ablation table as GitHub markdown (deltas: + = group helped)."""
    cols = ["label", "n_tickers", "mean_brier", "d_mean_brier", "ece", "d_ece",
            "big_move_auc", "d_big_move_auc", "coverage_abs_gap"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for c in cols:
            v = row.get(c)
            cells.append(f"{v:.5f}" if isinstance(v, float) else ("" if v is None else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
