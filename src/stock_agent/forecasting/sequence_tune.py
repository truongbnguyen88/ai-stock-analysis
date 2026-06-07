"""Full-scale LSTM tuning harness (Task 8, second attempt).

Addresses the spike's weaknesses — no early stopping, no calibration, tiny
universe, one config — with a proper protocol:

  * full 114-ticker universe;
  * **date-based train / val / test split** (leakage-safe: a training window's
    label must land at/before ``train_end``; val/test are strictly later);
  * **early stopping** on validation Brier (keeps the best epoch);
  * **temperature-scaling calibration** fit on the validation logits;
  * **random hyperparameter search**, ranked by calibrated val Brier;
  * the winner re-checked on a **disjoint held-out basket** via the existing
    walk-forward harness against the toolkit (anti-selection-bias).

Experimental / run-once: requires the ``[sequence]`` extra (torch). Invoke via
``run_lstm_tuning``. Not wired into the app. Determinism is per-config
(``manual_seed``) but multi-threaded for speed, so val Brier may wobble in the
last digits between machines; the held-out comparison is the decision number.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date as Date
from typing import Any

import numpy as np

from stock_agent.features.news_history import NewsStore
from stock_agent.features.price_features import PRICE_FEATURE_COLS
from stock_agent.forecasting.buckets import (
    assign_bucket,
    buckets_for_horizon,
    thresholds_for_horizon,
)
from stock_agent.forecasting.sequence import (
    SequenceModel,
    _build_net,
    _feature_frame,
    _Standardizer,
)
from stock_agent.logging_config import get_logger
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)


def _ticker_windows(
    series: PriceSeries, *, horizon: int, lookback: int, stride: int = 1,
    news_store: NewsStore | None = None, news_cols: list[str] | None = None, lag_days: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Date]]:
    """``(X[N,L,F], bucket[N], fwd[N], end_dates[N])`` for one ticker, leakage-safe.

    ``stride`` subsamples overlapping windows (every ``stride``-th day) to bound the
    pooled training size for the heavy nets — the windows stay leakage-safe. With
    ``news_store`` the feature dim ``F`` includes the leakage-safe news-history columns.
    """
    feats, _ = _feature_frame(series, news_store=news_store, news_cols=news_cols, lag_days=lag_days)
    closes = np.asarray(series.closes, dtype=float)
    dates = list(series.dates)
    n = len(feats)
    arr = feats.to_numpy(dtype=np.float32)
    buckets = buckets_for_horizon(horizon)
    X, yb, fwd, ed = [], [], [], []
    for t in range(lookback - 1, n - horizon, stride):
        r = closes[t + horizon] / closes[t] - 1.0
        if not np.isfinite(r):
            continue
        X.append(arr[t - lookback + 1 : t + 1])
        yb.append(assign_bucket(float(r), buckets))
        fwd.append(r)
        ed.append(dates[t])
    if not X:
        empty = np.empty((0, lookback, arr.shape[-1]), np.float32)
        return empty, np.empty(0, np.int64), np.empty(0, np.float64), []
    return np.stack(X), np.asarray(yb, np.int64), np.asarray(fwd, np.float64), ed


@dataclass
class _Split:
    Xtr: np.ndarray
    Ytr: np.ndarray
    ftr: np.ndarray  # train realized forward returns (for train-AUC diagnostic)
    Xva: np.ndarray
    fva: np.ndarray  # validation realized forward returns (for Brier)
    Yva: np.ndarray  # validation bucket labels (for NLL / temperature)
    scaler: _Standardizer
    n_features: int
    Xte: np.ndarray  # held-out TEST windows (val_end < end ≤ test_end)
    fte: np.ndarray  # test realized forward returns
    news_cols: list[str]  # the resolved news columns (empty == price-only)


def _assemble_split(
    series_list: list[PriceSeries], *, horizon: int, lookback: int, train_end: Date,
    val_end: Date, stride: int = 1, test_end: Date | None = None,
    news_store: NewsStore | None = None, lag_days: int = 1,
) -> _Split:
    """Pool windows into train (≤ train_end), val (≤ val_end), test (≤ test_end), by end-date.

    With ``news_store`` the news columns are resolved ONCE (off the first usable series) so
    every ticker — train, val, test — shares an identical feature layout.
    """
    news_cols: list[str] | None = None
    if news_store is not None:
        for s in series_list:
            _, all_cols = _feature_frame(s, news_store=news_store, lag_days=lag_days)
            news_cols = [c for c in all_cols if c not in PRICE_FEATURE_COLS]
            break

    Xtr, Ytr, ftr, Xva, fva, Yva, Xte, fte = [], [], [], [], [], [], [], []
    for s in series_list:
        X, yb, fwd, ed = _ticker_windows(
            s, horizon=horizon, lookback=lookback, stride=stride,
            news_store=news_store, news_cols=news_cols, lag_days=lag_days,
        )
        if not len(X):
            continue
        ed_arr = np.array(ed)
        for i in range(len(X)):
            end = ed_arr[i]
            if end <= train_end:
                Xtr.append(X[i])
                Ytr.append(yb[i])
                ftr.append(fwd[i])
            elif end <= val_end:
                Xva.append(X[i])
                fva.append(fwd[i])
                Yva.append(yb[i])
            elif test_end is None or end <= test_end:
                Xte.append(X[i])
                fte.append(fwd[i])
    if not Xtr or not Xva:
        raise ValueError("empty train or val split")
    Xtr_a = np.stack(Xtr)
    scaler = _Standardizer.fit(Xtr_a)
    n_feat = Xtr_a.shape[-1]
    Xte_s = scaler.transform(np.stack(Xte)) if Xte else np.empty((0, lookback, n_feat), np.float32)
    return _Split(
        Xtr=scaler.transform(Xtr_a),
        Ytr=np.asarray(Ytr, np.int64),
        ftr=np.asarray(ftr, np.float64),
        Xva=scaler.transform(np.stack(Xva)),
        fva=np.asarray(fva, np.float64),
        Yva=np.asarray(Yva, np.int64),
        scaler=scaler,
        n_features=n_feat,
        Xte=Xte_s,
        fte=np.asarray(fte, np.float64),
        news_cols=news_cols or [],
    )


def _exceedance(probs: np.ndarray, k: int) -> np.ndarray:
    """P(r > θ_k) = mass in buckets strictly above the k-th cut-point."""
    return np.asarray(probs[:, k + 1 :].sum(axis=1))


def _brier(probs: np.ndarray, fwd: np.ndarray, thresholds: list[float]) -> float:
    """Mean per-threshold exceedance Brier — the same target the backtest scores."""
    errs = []
    for k, thr in enumerate(thresholds):
        p = _exceedance(probs, k)
        y = (fwd > thr).astype(float)
        errs.append(float(np.mean((p - y) ** 2)))
    return float(np.mean(errs))


def _big_move_auc(probs: np.ndarray, fwd: np.ndarray, k: float) -> float | None:
    """ROC-AUC for P(|r| > k) — discrimination of the big-move signal.

    Buckets are ordered ``<-2k, [-2k,-k), [-k,0), [0,k), [k,2k), >=2k``; the inner two
    (indices 2,3) hold ``|r|<=k``, so the big-move probability is ``1 - (p2 + p3)``.
    """
    from sklearn.metrics import roc_auc_score

    pred = 1.0 - (probs[:, 2] + probs[:, 3])
    y = (np.abs(fwd) > k).astype(int)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, pred))


def _predict_probs(model: SequenceModel, X: np.ndarray) -> np.ndarray:
    """Temperature-scaled softmax probabilities for pre-scaled windows ``X``."""
    import torch

    net = _build_net(model.n_features, model.hidden, model.layers, model.dropout, model.layernorm)
    net.load_state_dict(model.state_dict)
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(X.astype(np.float32))).numpy() / model.temperature
    z = logits - logits.max(axis=1, keepdims=True)
    return np.asarray(np.exp(z) / np.exp(z).sum(axis=1, keepdims=True))


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Grid-search the temperature T minimising validation NLL (calibration)."""
    best_t, best_nll = 1.0, np.inf
    for t in np.linspace(0.5, 3.0, 26):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = float(-logp[np.arange(len(y)), y].mean())
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


@dataclass
class _Config:
    hidden: int
    layers: int
    lookback: int
    lr: float
    weight_decay: float
    dropout: float
    layernorm: bool = False
    batch_size: int = 512
    max_epochs: int = 40
    patience: int = 6


def _train_config(
    split: _Split, cfg: _Config, *, horizon: int, thresholds: list[float], seed: int
) -> tuple[SequenceModel, float, float | None, float | None]:
    """Early-stopped + temperature-calibrated training.

    Returns ``(model, calibrated val Brier, val big-move AUC, train big-move AUC)``.
    The train-vs-val AUC gap is the diagnostic: high train AUC + ~0.5 val AUC ⇒ pure
    overfit; ~0.5 on both ⇒ no extractable signal.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = _build_net(split.n_features, cfg.hidden, cfg.layers, cfg.dropout, cfg.layernorm)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(split.Xtr)
    yt = torch.from_numpy(split.Ytr)
    xva = torch.from_numpy(split.Xva)
    n = len(xt)
    gen = torch.Generator().manual_seed(seed)

    best_state: dict[str, Any] = {}
    best_brier, since_improve = np.inf, 0
    for _ in range(cfg.max_epochs):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            opt.zero_grad()
            loss_fn(net(xt[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vp = torch.softmax(net(xva), dim=1).numpy()
        vb = _brier(vp, split.fva, thresholds)
        if vb < best_brier - 1e-5:
            best_brier, since_improve = vb, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        val_logits = net(xva).numpy()
    temperature = _fit_temperature(val_logits, split.Yva)
    # calibrated val Brier (the ranking metric)
    z = val_logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    cal_probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    cal_brier = _brier(cal_probs, split.fva, thresholds)

    # Discrimination diagnostic at the inner big-move threshold k (smallest positive cut).
    k = thresholds[3]
    val_auc = _big_move_auc(cal_probs, split.fva, k)
    with torch.no_grad():
        train_probs = torch.softmax(net(torch.from_numpy(split.Xtr)), dim=1).numpy()
    train_auc = _big_move_auc(train_probs, split.ftr, k)

    model = SequenceModel(
        state_dict=best_state, scaler=split.scaler, horizon=horizon, lookback=cfg.lookback,
        hidden=cfg.hidden, layers=cfg.layers, n_features=split.n_features,
        dropout=cfg.dropout, temperature=temperature, layernorm=cfg.layernorm,
    )
    return model, cal_brier, val_auc, train_auc


def deep_config_grid() -> list[_Config]:
    """A careful DEEP-LSTM grid (3–4 layers, wide hidden, LayerNorm + dropout).

    LayerNorm + dropout + early stopping are what make the deep stacks trainable on the
    short news-covered window; the sweep is ranked by calibrated validation Brier.
    """
    # lr 3e-4 + dropout 0.2: gentler than 1e-3/0.3 (deep LSTMs converge better at low lr);
    # the lower lr gets more epochs/patience so early-stopping doesn't cut it short.
    out: list[_Config] = []
    for hidden in (128, 256):
        for layers in (3, 4):
            out.append(_Config(
                hidden=hidden, layers=layers, lookback=60, lr=3e-4, weight_decay=1e-4,
                dropout=0.2, layernorm=True, batch_size=512, max_epochs=100, patience=10,
            ))
    # one wider-lookback deep variant (same lr/dropout)
    out.append(_Config(
        hidden=192, layers=3, lookback=90, lr=3e-4, weight_decay=1e-4,
        dropout=0.2, layernorm=True, batch_size=512, max_epochs=100, patience=10,
    ))
    return out


def _sample_configs(n: int, seed: int) -> list[_Config]:
    rng = np.random.default_rng(seed)

    def pick(opts: list[float]) -> float:
        return opts[int(rng.integers(len(opts)))]

    seen: set[tuple[Any, ...]] = set()
    out: list[_Config] = []
    while len(out) < n and len(seen) < 500:
        c = _Config(
            hidden=int(pick([32, 64, 128])),
            layers=int(pick([1, 2])),
            lookback=int(pick([40, 60])),
            lr=pick([3e-4, 1e-3, 3e-3]),
            weight_decay=pick([0.0, 1e-5, 1e-4]),
            dropout=pick([0.1, 0.3, 0.5]),
        )
        key = (c.hidden, c.layers, c.lookback, c.lr, c.weight_decay, c.dropout)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


_TuneRow = tuple[_Config, float, "float | None", "float | None", SequenceModel]


def run_lstm_tuning(
    series_list: list[PriceSeries], *, horizon: int = 20, configs: list[_Config] | None = None,
    n_configs: int = 14, seed: int = 42, stride: int = 1,
) -> list[_TuneRow]:
    """Search configs; return ``[(config, val_brier, val_auc, train_auc, model)]`` ranked.

    Pass explicit ``configs`` (e.g. the heavy grid) or let it random-sample. ``stride``
    subsamples training windows to bound the heavy nets' compute.
    """
    all_dates = sorted({d for s in series_list for d in s.dates})
    train_end = all_dates[int(len(all_dates) * 0.60)]
    val_end = all_dates[int(len(all_dates) * 0.80)]
    log.info("lstm_tune.split", train_end=str(train_end), val_end=str(val_end), stride=stride)

    configs = configs or _sample_configs(n_configs, seed)
    split_cache: dict[int, _Split] = {}
    thresholds = thresholds_for_horizon(horizon)
    results: list[_TuneRow] = []
    for j, cfg in enumerate(configs):
        if cfg.lookback not in split_cache:
            split_cache[cfg.lookback] = _assemble_split(
                series_list, horizon=horizon, lookback=cfg.lookback,
                train_end=train_end, val_end=val_end, stride=stride,
            )
        t0 = time.time()
        model, vb, vauc, tauc = _train_config(
            split_cache[cfg.lookback], cfg, horizon=horizon, thresholds=thresholds, seed=seed
        )
        va = f"{vauc:.3f}" if vauc is not None else "n/a"
        ta = f"{tauc:.3f}" if tauc is not None else "n/a"
        print(
            f"  [{j+1:>2}/{len(configs)}] h{cfg.hidden} L{cfg.layers} ln{int(cfg.layernorm)} "
            f"lb{cfg.lookback} lr{cfg.lr:.0e} wd{cfg.weight_decay:.0e} do{cfg.dropout} "
            f"T{model.temperature:.2f}  -> valBrier {vb:.4f}  valAUC {va}  trainAUC {ta}  "
            f"({time.time()-t0:.0f}s)",
            flush=True,
        )
        results.append((cfg, vb, vauc, tauc, model))
    results.sort(key=lambda r: r[1])
    return results
