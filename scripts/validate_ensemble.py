"""Walk-forward validation: ensemble (equal vs skill-weighted) vs its members.

Skill-weighting = **online convex stacking**: at each fold the member weights are the
convex combination (simplex, ridge-pulled to uniform) that minimised the ensemble's
exceedance Brier on ALL PRIOR folds — never the current/future fold (leakage-safe).
Plain 1/Brier is too flat to matter here (members' Briers cluster ~0.17), so stacking
is the principled choice. Everything is scored with the production harness primitives
(walk_forward_splits, exceedance_probabilities, threshold_metrics, calibration_report)
on identical as-ofs, with per-fold pooled ML retrain (shared cache).

Run: PYTHONPATH=src python scripts/validate_ensemble.py
"""

from __future__ import annotations

import warnings
from datetime import date as Date
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from stock_agent.backtesting.calibration import calibration_report
from stock_agent.backtesting.metrics import threshold_metrics
from stock_agent.backtesting.runner import exceedance_probabilities
from stock_agent.backtesting.splitter import walk_forward_splits
from stock_agent.data.loader import PriceLoader
from stock_agent.data.market_context import fetch_vix
from stock_agent.forecasting.buckets import thresholds_for_horizon
from stock_agent.forecasting.ensemble import blend_forecasts
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.forecasting.ml import MLForecaster
from stock_agent.forecasting.monte_carlo import MonteCarlo
from stock_agent.forecasting.pooled import PooledModel, train_pooled_from_series
from stock_agent.forecasting.train_pooled import (
    fetch_universe_earnings,
    fetch_universe_series,
    load_universe,
)
from stock_agent.providers.registry import build_default_registry
from stock_agent.schemas.market import PriceSeries
from stock_agent.settings import get_settings

warnings.filterwarnings("ignore")

BASKET = ["NVDA", "AMD", "MSFT", "KO"]
HORIZONS = [20, 60]
TEST_SIZE = 6
MC_PATHS = 2000
UNIVERSE = "configs/universe.txt"
MEMBERS = ["historical_sim", "monte_carlo_bootstrap", "monte_carlo_garch", "logistic", "lightgbm"]
MODELS = [*MEMBERS, "ensemble_equal", "ensemble_skill"]
MIN_STACK = 40  # prior OOS samples before stacking kicks in (else equal weights)
RIDGE = 0.3  # pull stacked weights toward uniform (few folds → overfit guard)


class Trainer:
    def __init__(self) -> None:
        self.registry = build_default_registry(get_settings())
        self.universe = fetch_universe_series(load_universe(Path(UNIVERSE)), self.registry)
        self.earnings = fetch_universe_earnings([s.ticker for s in self.universe], self.registry)
        span = [d for s in self.universe for d in (s.dates[0], s.dates[-1])]
        vix = fetch_vix(self.registry, start=min(span), end=max(span)) if span else None
        self.vix = vix if (vix is not None and not vix.empty) else None
        self._cache: dict[tuple[str, int, Date], PooledModel] = {}

    def ml(self, model_type: str, horizon: int, train_end: Date) -> MLForecaster:
        key = (model_type, horizon, train_end)
        if key not in self._cache:
            sliced = [
                PriceSeries(ticker=s.ticker, bars=[b for b in s.bars if b.date <= train_end])
                for s in self.universe
            ]
            sliced = [s for s in sliced if len(s) >= 60]
            self._cache[key] = train_pooled_from_series(
                sliced, horizon_days=horizon, model_type=model_type,  # type: ignore[arg-type]
                earnings_by_ticker=self.earnings, vix=self.vix, calibrate=True,
            )
        return MLForecaster(model_type, model=self._cache[key], registry=self.registry)  # type: ignore[arg-type]


def _members(horizon: int, train_end: Date, trainer: Trainer) -> list:  # type: ignore[type-arg]
    return [
        HistoricalSimulation(),
        MonteCarlo(variant="bootstrap", n_paths=MC_PATHS),
        MonteCarlo(variant="garch", n_paths=MC_PATHS),
        trainer.ml("logistic", horizon, train_end),
        trainer.ml("lightgbm", horizon, train_end),
    ]


def stack_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convex (simplex) weights minimising ridge-regularised exceedance Brier on prior folds."""
    m = P.shape[1]
    u = np.full(m, 1.0 / m)

    def obj(w: np.ndarray) -> float:
        r = P @ w - y
        return float(np.mean(r * r) + RIDGE * np.mean((w - u) ** 2))

    res = minimize(
        obj, u, method="SLSQP", bounds=[(0.0, 1.0)] * m,
        constraints=[{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}],
    )
    w = np.clip(np.asarray(res.x, dtype=float), 0.0, None)
    return w / w.sum() if w.sum() > 0 else u


def _new_acc() -> dict:  # type: ignore[type-arg]
    return {"probs": None, "labels": None, "bm_p": [], "bm_y": [], "cp": [], "cy": []}


def _record(acc: dict, ex: list[float], labels: list[float], bm_p: float, bm_y: float) -> None:  # type: ignore[type-arg]
    if acc["probs"] is None:
        acc["probs"] = [[] for _ in ex]
        acc["labels"] = [[] for _ in ex]
    for k, (p, lab) in enumerate(zip(ex, labels, strict=True)):
        acc["probs"][k].append(p)
        acc["labels"][k].append(lab)
        acc["cp"].append(p)
        acc["cy"].append(lab)
    acc["bm_p"].append(bm_p)
    acc["bm_y"].append(bm_y)


def _finalize(acc: dict) -> tuple[float, float, float]:  # type: ignore[type-arg]
    per = [threshold_metrics(acc["labels"][k], acc["probs"][k], threshold=0.0) for k in range(len(acc["probs"]))]
    brier = float(np.mean([m.brier for m in per]))
    ece = calibration_report(acc["cp"], acc["cy"], n_bins=10).ece
    bm = threshold_metrics(acc["bm_y"], acc["bm_p"], threshold=0.0)
    return brier, ece, (bm.roc_auc if bm.roc_auc is not None else float("nan"))


def validate(ticker: str, horizon: int, trainer: Trainer, loader: PriceLoader):  # type: ignore[no-untyped-def]
    thr = thresholds_for_horizon(horizon)
    k_big = min(abs(t) for t in thr if t != 0)
    series = loader.load_recent(ticker, 2200, min_bars=252 + 2 * horizon).series
    closes = np.asarray(series.closes, dtype=float)
    dates = series.dates
    folds = walk_forward_splits(n_bars=len(series), horizon=horizon, min_train=252, test_size=TEST_SIZE)

    acc = {m: _new_acc() for m in MODELS}
    pool_P: list[list[float]] = []
    pool_y: list[float] = []
    w_skill = np.full(len(MEMBERS), 1.0 / len(MEMBERS))

    for fold in folds:
        models = _members(horizon, dates[fold.train_end], trainer)
        fold_P: list[list[float]] = []
        fold_y: list[float] = []
        for t in fold.test_as_of:
            sub = PriceSeries(ticker=series.ticker, bars=series.bars[: t + 1])
            try:
                fcs = [mdl.forecast(sub, horizon_days=horizon, as_of=dates[t]) for mdl in models]
            except (ValueError, RuntimeError):
                continue  # skip as-of if ANY member fails (keeps models on identical samples)
            realized = float(closes[t + horizon] / closes[t] - 1.0)
            labels = [1.0 if realized > theta else 0.0 for theta in thr]
            bm_y = 1.0 if abs(realized) > k_big else 0.0
            member_ex = [exceedance_probabilities(fc, thr) for fc in fcs]

            ens_eq = blend_forecasts(fcs, None, ticker=ticker, as_of=dates[t], horizon_days=horizon)
            ens_sk = blend_forecasts(fcs, list(w_skill), ticker=ticker, as_of=dates[t], horizon_days=horizon)
            allfc = dict(zip(MEMBERS, fcs, strict=True))
            allfc["ensemble_equal"], allfc["ensemble_skill"] = ens_eq, ens_sk
            for name, fc in allfc.items():
                ex = exceedance_probabilities(fc, thr)
                bm_p = fc.buckets[0].probability + fc.buckets[-1].probability
                _record(acc[name], ex, labels, bm_p, bm_y)
            for k in range(len(thr)):  # stacking rows: one per (as_of, threshold)
                fold_P.append([member_ex[mi][k] for mi in range(len(MEMBERS))])
                fold_y.append(labels[k])

        pool_P.extend(fold_P)
        pool_y.extend(fold_y)
        if len(pool_y) >= MIN_STACK:
            w_skill = stack_weights(np.asarray(pool_P), np.asarray(pool_y))

    return {m: _finalize(acc[m]) for m in MODELS}, w_skill


def main() -> None:
    trainer = Trainer()
    loader = PriceLoader(trainer.registry)
    agg: dict[tuple[int, str], list[tuple[float, float, float]]] = {}

    for horizon in HORIZONS:
        for ticker in BASKET:
            res, w = validate(ticker, horizon, trainer, loader)
            print(f"\n=== {ticker} h{horizon} ===   skill weights: "
                  + ", ".join(f"{n.split('_')[0][:4]} {wi:.2f}" for n, wi in zip(MEMBERS, w, strict=True)))
            print(f"  {'model':>16} | {'Brier':>8} {'ECE':>7} {'bigAUC':>7}")
            for m in MODELS:
                b, e, a = res[m]
                print(f"  {m:>16} | {b:8.4f} {e:7.4f} {a:7.3f}")
                agg.setdefault((horizon, m), []).append((b, e, a))

    print(f"\n{'=' * 56}\nAGGREGATE (mean over {len(BASKET)} tickers)\n{'=' * 56}")
    for horizon in HORIZONS:
        print(f"\n  h{horizon}:  {'model':>16} | {'Brier':>8} {'ECE':>7} {'bigAUC':>7}")
        rows = {m: np.array(agg[(horizon, m)]) for m in MODELS}
        best = min(MEMBERS, key=lambda m: rows[m][:, 0].mean())
        for m in MODELS:
            b, e, a = rows[m].mean(axis=0)
            tag = ""
            if m == best:
                tag = " *best member*"
            elif m == "ensemble_skill":
                tag = " <== skill ensemble"
            print(f"        {m:>16} | {b:8.4f} {e:7.4f} {a:7.3f}{tag}")


if __name__ == "__main__":
    main()
