"""Append-only retrieval telemetry writer (advanced-RAG A6.1a) — default-OFF, $0, side-effect only.

Persists ``RetrievalLogEntry`` rows as **JSONL** (one JSON object per line) under
``settings.retrieval_log_dir``. The whole module is gated by ``settings.retrieval_logging``: when
off (the default), ``log_retrieval`` is a **no-op** that touches no disk — so the read path is
byte-identical to today until logging is explicitly enabled. Nothing here reads or influences
retrieval; it only records decisions for later off-policy learning (A6.1d–f).

JSONL (not one big JSON array) is deliberate: append is O(1), a crash mid-write costs at most the
last line, and the file streams row-by-row for large logs. ``read_log`` tolerates blank lines and
skips malformed rows (a partial final write) rather than failing the whole load.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from stock_agent.schemas.retrieval_log import RetrievalLogEntry
from stock_agent.settings import Settings

log = structlog.get_logger(__name__)

_DEFAULT_FILENAME = "retrieval_log.jsonl"


def log_retrieval(
    entry: RetrievalLogEntry,
    *,
    settings: Settings,
    filename: str = _DEFAULT_FILENAME,
) -> Path | None:
    """Append ``entry`` to the JSONL log iff ``settings.retrieval_logging`` is on; else no-op.

    Returns the log file path when a row was written, ``None`` when logging is disabled (the
    default). Creates ``settings.retrieval_log_dir`` on first write. One line per call, flushed on
    close — safe for concurrent appends of whole lines on POSIX.
    """
    if not settings.retrieval_logging:
        return None
    log_dir = Path(settings.retrieval_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / filename
    # model_dump_json => one compact line; append mode keeps the file append-only.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry.model_dump_json() + "\n")
    return path


def read_log(path: Path) -> list[RetrievalLogEntry]:
    """Load all entries from a JSONL retrieval log; skip blank/malformed lines (partial writes).

    Returns ``[]`` if the file does not exist (logging never ran). A malformed final line — the
    only realistic corruption for append-only JSONL — is logged and skipped, not raised, so an
    interrupted run's log is still usable.
    """
    path = Path(path)
    if not path.exists():
        return []
    entries: list[RetrievalLogEntry] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entries.append(RetrievalLogEntry.model_validate_json(stripped))
        except ValueError:
            log.warning("retrieval_log.malformed_line", path=str(path), line_no=i)
    return entries
