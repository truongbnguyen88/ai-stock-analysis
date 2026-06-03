"""Experimental pooled LSTM sequence forecaster (Task 8 spike).

Tests whether *temporal structure* — vol clustering, momentum/mean-reversion
dynamics that the tabular one-day feature snapshot flattens — adds out-of-sample
skill beyond the toolkit. A small LSTM consumes a sequence of the existing
scale-free daily features and predicts a **softmax over the 6 horizon-scaled
return buckets**, which is a native ``ScenarioForecast`` (so it drops into the
backtest harness alongside every other model).

Leakage discipline: features at day ``t`` use only data ≤ ``t``; the bucket label
uses ``P_{t+h}`` and any window whose label reaches beyond the training cutoff is
dropped; the per-feature standardizer is fit on the training pool only. The
trained model must be used **only** on as-of dates after its training cutoff.

Determinism: all RNGs seeded, single-threaded CPU,
``torch.use_deterministic_algorithms(True)`` — reruns are bit-stable.

EXPERIMENTAL — requires the ``[sequence]`` extra (torch). Reachable via the
evaluation helpers / a wrapped ``SequenceForecaster``; deliberately NOT wired into
the agent, written report, CI, or the default backtest set. Promote only if it
beats the toolkit on Brier/ECE OOS (docs/TASKS.md Task 8).
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from datetime import date as Date
from typing import Any

import numpy as np

from stock_agent.features.price_features import PRICE_FEATURE_COLS, build_price_feature_matrix
from stock_agent.forecasting.buckets import assign_bucket, buckets_for_horizon, make_prob_buckets
from stock_agent.forecasting.historical import HistoricalSimulation
from stock_agent.forecasting.ml import _bucket_midpoint
from stock_agent.logging_config import get_logger
from stock_agent.schemas.forecast import ScenarioForecast
from stock_agent.schemas.market import PriceSeries

log = get_logger(__name__)

_MODEL_NAME = "lstm_seq"
_LOOKBACK = 60  # trading-day sequence length fed to the LSTM
_N_BUCKETS = 6


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _ticker_sequences(
    series: PriceSeries, *, horizon: int, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Leakage-safe ``(X[N, L, F], y[N])`` for one ticker.

    Each window ends at day ``t`` (features ≤ ``t``); the label is the bucket of
    the realized forward ``h``-day return at ``t`` (so ``t + h`` must exist). The
    final ``horizon`` bars have no label and are skipped.
    """
    feats = build_price_feature_matrix(series)[PRICE_FEATURE_COLS]
    closes = np.asarray(series.closes, dtype=float)
    n = len(feats)
    fwd = np.full(n, np.nan)
    if n > horizon:
        fwd[:-horizon] = closes[horizon:] / closes[:-horizon] - 1.0
    arr = feats.to_numpy(dtype=np.float32)
    buckets = buckets_for_horizon(horizon)

    seqs: list[np.ndarray] = []
    labels: list[int] = []
    for t in range(lookback - 1, n):
        if not np.isfinite(fwd[t]):  # label reaches past available history
            continue
        seqs.append(arr[t - lookback + 1 : t + 1])
        labels.append(assign_bucket(float(fwd[t]), buckets))
    if not seqs:
        return np.empty((0, lookback, len(PRICE_FEATURE_COLS)), np.float32), np.empty(0, np.int64)
    return np.stack(seqs), np.asarray(labels, dtype=np.int64)


@dataclass
class _Standardizer:
    """Per-feature mean/std fit on the training pool; NaN → 0 after scaling."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> _Standardizer:
        flat = x.reshape(-1, x.shape[-1])
        with warnings.catch_warnings():  # all-NaN feature cols (e.g. VIX/earnings) → handled below
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(flat, axis=0)
            std = np.nanstd(flat, axis=0)
        std[~np.isfinite(std) | (std == 0)] = 1.0
        mean[~np.isfinite(mean)] = 0.0
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        scaled = np.nan_to_num((x - self.mean) / self.std, nan=0.0)
        return np.asarray(scaled, dtype=np.float32)


@dataclass
class SequenceModel:
    """Trained LSTM weights + standardizer + config (one horizon).

    ``temperature`` (>1 softens, <1 sharpens) is a post-hoc calibration scalar fit
    on a validation slice; ``dropout`` is stored so the net can be reconstructed.
    """

    state_dict: dict[str, Any]
    scaler: _Standardizer
    horizon: int
    lookback: int
    hidden: int
    layers: int
    n_features: int
    dropout: float = 0.0
    temperature: float = 1.0
    layernorm: bool = False


def _build_net(
    n_features: int, hidden: int, layers: int, dropout: float = 0.0, layernorm: bool = False
) -> Any:
    from torch import nn

    class LSTMNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                n_features, hidden, num_layers=layers, batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
            # LayerNorm stabilises the deeper (3–4 layer) stacks in the tuning sweep.
            self.norm = nn.LayerNorm(hidden) if layernorm else nn.Identity()
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, _N_BUCKETS)

        def forward(self, x: Any) -> Any:
            out, _ = self.lstm(x)
            return self.head(self.drop(self.norm(out[:, -1, :])))  # last step → bucket logits

    return LSTMNet()


def train_sequence_model(
    series_list: list[PriceSeries],
    *,
    horizon: int,
    lookback: int = _LOOKBACK,
    hidden: int = 48,
    layers: int = 1,
    epochs: int = 25,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 42,
    dropout: float = 0.0,
    weight_decay: float = 0.0,
    layernorm: bool = False,
) -> SequenceModel:
    """Train a pooled LSTM over the universe's leakage-safe feature sequences.

    Fixed-epoch trainer (no early stopping) used by the unit tests and simple
    callers; the tuning harness (``sequence_tune``) does its own early-stopped,
    validation-selected, temperature-calibrated training.
    """
    import torch
    from torch import nn

    _seed_everything(seed)
    xs, ys = [], []
    for s in series_list:
        try:
            x, y = _ticker_sequences(s, horizon=horizon, lookback=lookback)
        except ValueError:
            continue
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError("no training sequences built from the universe")
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    scaler = _Standardizer.fit(X)
    Xs = scaler.transform(X)
    log.info("sequence.train_start", n_seq=len(Xs), horizon=horizon, features=X.shape[-1])

    net = _build_net(X.shape[-1], hidden, layers, dropout, layernorm)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(Y)
    n = len(xt)
    gen = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return SequenceModel(
        state_dict={k: v.clone() for k, v in net.state_dict().items()},
        scaler=scaler,
        horizon=horizon,
        lookback=lookback,
        hidden=hidden,
        layers=layers,
        n_features=X.shape[-1],
        dropout=dropout,
        layernorm=layernorm,
    )


class SequenceForecaster:
    """``ForecastModel`` wrapping a trained :class:`SequenceModel` (one horizon)."""

    def __init__(self, model: SequenceModel) -> None:
        self.name = _MODEL_NAME
        self._model = model
        self._net: Any = None  # lazily reconstructed torch module

    def _net_eval(self) -> tuple[Any, Any]:
        import torch

        if self._net is None:
            net = _build_net(
                self._model.n_features, self._model.hidden, self._model.layers,
                self._model.dropout, self._model.layernorm,
            )
            net.load_state_dict(self._model.state_dict)
            net.eval()
            self._net = net
        return self._net, torch

    def forecast(
        self, series: PriceSeries, *, horizon_days: int, as_of: Date | None = None
    ) -> ScenarioForecast:
        as_of = as_of or series.bars[-1].date
        if horizon_days != self._model.horizon:
            raise ValueError(
                f"model trained for horizon {self._model.horizon}, asked {horizon_days}"
            )
        feats = build_price_feature_matrix(series)[PRICE_FEATURE_COLS]
        if len(feats) < self._model.lookback:
            return self._fallback(series, horizon_days, as_of)

        window = feats.to_numpy(dtype=np.float32)[-self._model.lookback :]
        x = self._model.scaler.transform(window)[None]  # [1, L, F]
        net, torch = self._net_eval()
        with torch.no_grad():
            logits = net(torch.from_numpy(x)) / self._model.temperature  # temp-scaled calibration
            probs = torch.softmax(logits, dim=1).numpy()[0]

        buckets = make_prob_buckets([float(p) for p in probs], buckets_for_horizon(horizon_days))
        expected = sum(_bucket_midpoint(b) * b.probability for b in buckets)
        return ScenarioForecast(
            ticker=series.ticker,
            as_of=as_of,
            horizon_days=horizon_days,
            model_name=self.name,
            buckets=buckets,
            expected_return=float(expected),
            upside_prob=sum(b.probability for b in buckets if b.lower is not None and b.lower >= 0),
            downside_prob=sum(
                b.probability for b in buckets if b.upper is not None and b.upper <= 0
            ),
            calibration_status="uncalibrated",
            notes=f"Pooled LSTM (lookback {self._model.lookback}); experimental.",
        )

    def _fallback(
        self, series: PriceSeries, horizon_days: int, as_of: Date
    ) -> ScenarioForecast:
        fc = HistoricalSimulation().forecast(series, horizon_days=horizon_days, as_of=as_of)
        note = "LSTM fallback (insufficient history); used unconditional historical_sim."
        return fc.model_copy(update={"model_name": self.name, "notes": note})
