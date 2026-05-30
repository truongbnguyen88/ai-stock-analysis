"""Offline pooled-training orchestration: universe -> fetch -> train -> persist.

Reads a ticker universe, fetches deep price history for each (via the provider
registry / cache), and delegates to ``train_pooled_from_series``. Network-bound
and slow on a cold cache (one download per ticker), but a one-time step — the
resulting artifact is loaded cheaply at inference.
"""

from __future__ import annotations

from pathlib import Path

from stock_agent.data.loader import PriceLoader
from stock_agent.forecasting.pooled import ModelType, PooledModel, train_pooled_from_series
from stock_agent.logging_config import get_logger
from stock_agent.providers.base import ProviderError
from stock_agent.providers.registry import ProviderRegistry, build_default_registry
from stock_agent.schemas.market import PriceSeries
from stock_agent.settings import Settings

log = get_logger(__name__)

# ~6 years of calendar days: enough trading bars for warmup (MA200) + many rows.
_TRAIN_LOOKBACK_DAYS = 2200


def load_universe(path: Path) -> list[str]:
    """Read tickers from a universe file (one per line; '#' comments, blanks ignored)."""
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tickers.append(line.upper())
    return tickers


def fetch_universe_series(
    tickers: list[str], registry: ProviderRegistry, *, lookback_days: int = _TRAIN_LOOKBACK_DAYS
) -> list[PriceSeries]:
    """Fetch price history for each ticker; skip any that fail (logged)."""
    loader = PriceLoader(registry)
    out: list[PriceSeries] = []
    for ticker in tickers:
        try:
            out.append(loader.load_recent(ticker, lookback_days, min_bars=200).series)
        except ProviderError as exc:
            log.warning("train.fetch_failed", ticker=ticker, error=str(exc))
    return out


def train_pooled(
    universe_path: Path,
    settings: Settings,
    *,
    model_type: ModelType = "xgboost",
    horizon_days: int = 20,
    registry: ProviderRegistry | None = None,
) -> tuple[PooledModel, Path]:
    """Train a pooled model over the universe and persist it. Returns (model, path)."""
    from stock_agent.forecasting.pooled import default_model_path

    registry = registry or build_default_registry(settings)
    tickers = load_universe(universe_path)
    log.info("train.start", tickers=len(tickers), model=model_type, horizon=horizon_days)

    series_list = fetch_universe_series(tickers, registry)
    if not series_list:
        raise ValueError("no price series fetched for the universe")

    model = train_pooled_from_series(series_list, horizon_days=horizon_days, model_type=model_type)

    path = default_model_path(Path(settings.output_dir) / "models", model_type, horizon_days)
    model.save(path)
    return model, path
