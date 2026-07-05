"""A6.1a — retrieval telemetry writer: round-trip, off-is-noop, append accumulates (offline, CI)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from stock_agent.rag.retrieval_log import log_retrieval, read_log
from stock_agent.schemas.retrieval_log import RetrievalLogEntry, ScoredChunkRef
from stock_agent.settings import Settings


def _entry(action: str = "hybrid", *, query: str = "what are NVDA AI risks?") -> RetrievalLogEntry:
    return RetrievalLogEntry(
        timestamp=datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC),
        query=query,
        ticker="NVDA",
        context={"n_tokens": 5.0, "is_bridging": 0.0, "in_graph_universe": 1.0},
        action=action,
        propensity=0.2,  # uniform over 5 arms
        retrieved=[ScoredChunkRef(chunk_id="NVDA:10-K:2025-02-26:3", score=0.71)],
        seed=42,
    )


def _on(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, retrieval_logging=True, retrieval_log_dir=tmp_path)


def test_off_is_noop(tmp_path: Path) -> None:
    """Default-OFF: no file created, returns None, byte-identical no-op."""
    settings = Settings(_env_file=None, retrieval_log_dir=tmp_path)  # logging defaults to False
    assert settings.retrieval_logging is False
    assert log_retrieval(_entry(), settings=settings) is None
    assert list(tmp_path.iterdir()) == []


def test_round_trip(tmp_path: Path) -> None:
    """A written entry reloads field-for-field (context map, propensity, refs, seed)."""
    settings = _on(tmp_path)
    path = log_retrieval(_entry(), settings=settings)
    assert path is not None and path.exists()
    loaded = read_log(path)
    assert len(loaded) == 1
    e = loaded[0]
    assert e.action == "hybrid"
    assert e.propensity == 0.2
    assert e.context == {"n_tokens": 5.0, "is_bridging": 0.0, "in_graph_universe": 1.0}
    assert e.retrieved[0].chunk_id == "NVDA:10-K:2025-02-26:3"
    assert e.seed == 42


def test_append_accumulates(tmp_path: Path) -> None:
    """Repeated logging appends (JSONL), preserving order; no overwrite."""
    settings = _on(tmp_path)
    for arm in ("dense", "hybrid", "graph"):
        log_retrieval(_entry(action=arm), settings=settings)
    loaded = read_log(tmp_path / "retrieval_log.jsonl")
    assert [e.action for e in loaded] == ["dense", "hybrid", "graph"]


def test_read_missing_and_malformed(tmp_path: Path) -> None:
    """Missing file -> []; a malformed/partial final line is skipped, not raised."""
    assert read_log(tmp_path / "nope.jsonl") == []
    settings = _on(tmp_path)
    log_retrieval(_entry(action="dense"), settings=settings)
    path = tmp_path / "retrieval_log.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n")  # blank line (tolerated)
        fh.write('{"query": "partial",')  # truncated write (skipped)
    loaded = read_log(path)
    assert len(loaded) == 1 and loaded[0].action == "dense"


def test_deterministic_logger_propensity_none(tmp_path: Path) -> None:
    """A deterministic logging policy records propensity=None (OPE will skip such rows)."""
    settings = _on(tmp_path)
    e = _entry()
    e.propensity = None
    path = log_retrieval(e, settings=settings)
    assert path is not None
    assert read_log(path)[0].propensity is None
