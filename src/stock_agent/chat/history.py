"""Persistent chat-thread store for the Streamlit frontend.

Each thread is one JSON file (``{id}.json``) under a configurable directory. A
thread stores both the display messages (text + already-serialized charts) AND
the ``agent_history`` (Anthropic-format messages) so reopening it resumes the
conversation with full context. Threads older than the retention window (by
``updated_at``) are pruned on demand.

Pure and dependency-light (stdlib + pathlib); the store is viz-agnostic — the UI
serializes ``ChartSpec``s to dicts before saving and rebuilds them on load.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_thread_id() -> str:
    return uuid.uuid4().hex


def derive_title(display_messages: list[dict[str, Any]], *, max_len: int = 60) -> str:
    """A thread title from the first user message (ChatGPT-style), trimmed."""
    for m in display_messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            text = " ".join(m["content"].split())
            if text:
                return text if len(text) <= max_len else text[: max_len - 1] + "…"
    return "New chat"


def to_jsonable(obj: Any) -> Any:
    """Recursively coerce to JSON-safe data; Anthropic SDK blocks -> dicts.

    ``agent_history`` may contain SDK content-block objects (pydantic models with
    ``model_dump``); normalize them so the thread serializes and the dicts remain
    valid Anthropic message content when the conversation is resumed.
    """
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump())
    return str(obj)  # last resort — keeps a thread saveable rather than crashing


@dataclass(frozen=True)
class ThreadMeta:
    """Lightweight thread descriptor for the sidebar list."""

    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class ChatThread:
    id: str
    title: str
    created_at: str
    updated_at: str
    display_messages: list[dict[str, Any]]  # [{role, content, charts: [chart_dict]}]
    agent_history: list[dict[str, Any]]  # JSON-safe Anthropic messages (resumable)

    @property
    def meta(self) -> ThreadMeta:
        return ThreadMeta(self.id, self.title, self.created_at, self.updated_at)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "display_messages": to_jsonable(self.display_messages),
            "agent_history": to_jsonable(self.agent_history),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ChatThread:
        return cls(
            id=d["id"],
            title=d["title"],
            created_at=d["created_at"],
            updated_at=d["updated_at"],
            display_messages=d.get("display_messages", []),
            agent_history=d.get("agent_history", []),
        )


class ChatStore:
    """JSON-file-per-thread store with age-based pruning."""

    def __init__(self, directory: Path | str, *, retention_days: int = 30) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days

    def _path(self, thread_id: str) -> Path:
        return self._dir / f"{thread_id}.json"

    def save(self, thread: ChatThread) -> None:
        self._path(thread.id).write_text(
            json.dumps(thread.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, thread_id: str) -> ChatThread:
        return ChatThread.from_json(
            json.loads(self._path(thread_id).read_text(encoding="utf-8"))
        )

    def delete(self, thread_id: str) -> None:
        self._path(thread_id).unlink(missing_ok=True)

    def list_threads(self) -> list[ThreadMeta]:
        """Thread descriptors, most-recently-updated first (corrupt files skipped)."""
        metas: list[ThreadMeta] = []
        for path in self._dir.glob("*.json"):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                metas.append(ThreadMeta(d["id"], d["title"], d["created_at"], d["updated_at"]))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas

    def prune(self) -> int:
        """Delete threads older than the retention window; return how many were removed."""
        cutoff = datetime.now(UTC) - timedelta(days=self._retention_days)
        removed = 0
        for path in self._dir.glob("*.json"):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                if datetime.fromisoformat(d["updated_at"]) < cutoff:
                    path.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError, OSError, ValueError):
                continue
        return removed
