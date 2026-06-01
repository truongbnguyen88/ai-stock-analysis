"""CLI `train --all`: retrains the production toolkit and drops h5 (no network)."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from stock_agent.settings import Settings

# The stock_agent.cli package shadows the `app` submodule with the Typer instance.
app = importlib.import_module("stock_agent.cli.app")


def test_train_all_covers_toolkit_and_drops_h5(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    def fake_train_pooled(universe, settings, *, model_type, horizon_days, **kw):  # type: ignore[no-untyped-def]
        calls.append((model_type, horizon_days))
        path = tmp_path / "models" / f"pooled_{model_type}_h{horizon_days}.joblib"
        trained = SimpleNamespace(n_tickers=100, n_train_rows=50_000, classifiers={1: 1}, notes=[])
        return trained, path

    # Pre-create stale h5 artifacts that --all should remove.
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    for m in ("logistic", "lightgbm"):
        (models_dir / f"pooled_{m}_h5.joblib").write_bytes(b"stale")

    # The forecasting package shadows the train_pooled submodule with the function;
    # patch the real module object so the lazy `from ... import train_pooled` sees it.
    tp_module = importlib.import_module("stock_agent.forecasting.train_pooled")
    monkeypatch.setattr(tp_module, "train_pooled", fake_train_pooled)
    monkeypatch.setattr(app, "configure_logging", lambda s: None)
    monkeypatch.setattr(
        app, "get_settings", lambda: Settings(_env_file=None, output_dir=tmp_path)
    )

    universe = tmp_path / "universe.txt"
    universe.write_text("NVDA\nKO\n")

    app.train(universe=universe, train_all=True)

    # Trained exactly logistic + lightgbm at 20/30/60 (h5 NOT trained).
    assert set(calls) == {
        ("logistic", 20), ("logistic", 30), ("logistic", 60),
        ("lightgbm", 20), ("lightgbm", 30), ("lightgbm", 60),
    }
    # Stale h5 artifacts removed.
    assert not (models_dir / "pooled_logistic_h5.joblib").exists()
    assert not (models_dir / "pooled_lightgbm_h5.joblib").exists()
