"""Ticker normalization + universe loading for document downloads (RAG P1).

The boundary where tickers ENTER the RAG corpus. Tickers are normalized to
upper-case here so every downstream chunk's metadata filter (``ticker = NVDA``)
matches regardless of how the user typed it — the correctness fix flagged in the
P0 review. CIK resolution itself lives in ``providers/sec_edgar.py`` (it is an
EDGAR API call); this module only handles the local, pure concerns.
"""

from __future__ import annotations

from pathlib import Path


def normalize_ticker(ticker: str) -> str:
    """Canonical ticker form for storage + retrieval filters (trim + upper-case)."""
    return ticker.strip().upper()


def load_universe(path: Path) -> list[str]:
    """Read tickers from a universe file (one per line; ``#`` comments; blanks ignored).

    Kept local to ``documents`` (a 3-line reader) rather than importing the training
    module's loader, so the RAG layer does not depend on ``forecasting`` internals.
    """
    tickers: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            tickers.append(normalize_ticker(stripped))
    return tickers
