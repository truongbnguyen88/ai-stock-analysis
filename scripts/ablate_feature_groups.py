"""Walk-forward ablation of candidate feature groups (Tier 1).

Runs the pooled-ML backtest per ticker for the baseline feature set and for
baseline + each candidate group, then tabulates the out-of-sample deltas
(Brier / ECE / big-move AUC / interval-coverage gap). Only groups that improve
Brier without hurting calibration are flagged as promotion candidates.

This is a RUNTIME experiment (network + per-fold training) — not part of
``make check``. Metric aggregation/decision logic lives in
``stock_agent.backtesting.ablation`` and is unit-tested offline.

Usage:
  python scripts/ablate_feature_groups.py \
      --tickers NVDA MSFT AAPL JPM XOM --horizon 20 --model logistic \
      --groups volume high52w session shape relstr
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from stock_agent.backtesting.ablation import (
    ablation_table,
    is_promotion_candidate,
    render_markdown,
)
from stock_agent.features.price_features import FEATURE_GROUPS
from stock_agent.logging_config import get_logger
from stock_agent.pipelines.backtest import run_backtest_pipeline
from stock_agent.schemas.backtest import BacktestResult
from stock_agent.settings import get_settings

log = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--model", choices=["logistic", "lightgbm"], default="logistic")
    p.add_argument(
        "--groups", nargs="+", default=sorted(FEATURE_GROUPS),
        help="candidate groups to ablate (each compared individually vs baseline)",
    )
    p.add_argument("--min-train", type=int, default=252)
    p.add_argument("--test-size", type=int, default=6)
    p.add_argument(
        "--universe", type=Path, default=Path("configs/universe.txt"),
        help="universe file the pooled model trains on (per-fold)",
    )
    p.add_argument("--out", type=Path, default=None, help="JSON output path (default: outputs/)")
    return p.parse_args()


def _run_label(
    tickers: list[str], horizon: int, model: str, groups: list[str] | None,
    *, min_train: int, test_size: int, settings: object, universe: Path,
) -> list[BacktestResult]:
    """Backtest the ML model on each ticker for one feature-group selection."""
    results: list[BacktestResult] = []
    for ticker in tickers:
        out = run_backtest_pipeline(
            ticker, horizon, model_names=[model], settings=settings,  # type: ignore[arg-type]
            min_train=min_train, test_size=test_size, log_experiment=False,
            feature_groups=groups, universe_path=universe,
        )
        results.append(out[model])
    return results


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    by_label: dict[str, list[BacktestResult]] = {}

    log.info("ablation.baseline", tickers=args.tickers, universe=str(args.universe))
    by_label["baseline"] = _run_label(
        args.tickers, args.horizon, args.model, None,
        min_train=args.min_train, test_size=args.test_size, settings=settings,
        universe=args.universe,
    )
    for group in args.groups:
        log.info("ablation.group", group=group)
        by_label[group] = _run_label(
            args.tickers, args.horizon, args.model, [group],
            min_train=args.min_train, test_size=args.test_size, settings=settings,
            universe=args.universe,
        )

    rows = ablation_table(by_label)
    candidates = [r["label"] for r in rows if is_promotion_candidate(r)]

    print(render_markdown(rows))  # noqa: T201 — script output is the deliverable
    print(f"\nPromotion candidates (Brier↓ without ECE↑): {candidates or 'none'}")  # noqa: T201

    out_path = args.out or (
        Path(settings.output_dir) / "experiments"
        / f"ablation_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_h{args.horizon}_{args.model}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"config": vars(args) | {"tickers": args.tickers}, "rows": rows,
             "promotion_candidates": candidates},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    log.info("ablation.done", out=str(out_path), candidates=candidates)


if __name__ == "__main__":
    main()
