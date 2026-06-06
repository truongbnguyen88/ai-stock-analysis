"""Disk cache for provider responses.

Purpose: respect free-tier rate limits and make runs reproducible. A cache hit
costs zero API calls. Cached values are JSON strings (we serialize normalized
pydantic domain objects), keyed by a stable hash of the call parameters and
expired by file mtime against a TTL.

This is intentionally simple (filesystem, one file per key). It is not concurrent
or distributed; that is adequate for a single-user research CLI.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def make_key(*parts: object) -> str:
    """Build a stable cache key from arbitrary call parameters.

    Uses JSON with ``default=str`` so dates/enums serialize deterministically;
    order of parts is significant (it encodes provider + method + args).
    """
    payload = json.dumps(parts, default=str, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class DiskCache:
    """A TTL'd, file-backed cache of JSON strings."""

    def __init__(self, cache_dir: Path, ttl_seconds: int, *, enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str, *, ttl_override: int | None = None) -> str | None:
        """Return the cached value, or ``None`` if missing/expired/disabled.

        ``ttl_override`` lets a caller apply a shorter (or longer) freshness window than
        the cache default — e.g. news uses a short TTL for recency while prices keep the
        long default. ``None`` uses ``self.ttl_seconds``.
        """
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        # Expire by file age. TTL <= 0 means "never expire".
        ttl = self.ttl_seconds if ttl_override is None else ttl_override
        if ttl > 0 and (time.time() - path.stat().st_mtime) > ttl:
            return None
        return path.read_text(encoding="utf-8")

    def set(self, key: str, value: str) -> None:
        """Persist ``value`` under ``key`` (atomic write via temp + replace)."""
        if not self.enabled:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(key)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX; avoids torn reads


def cached_model(
    cache: DiskCache, key: str, model_cls: type[T], producer: Callable[[], T],
    *, ttl: int | None = None,
) -> T:
    """Return a cached model if present, else produce, cache, and return it.

    Centralizes the get -> deserialize / produce -> serialize -> set pattern so
    providers don't repeat it. ``producer`` performs the actual (cache-miss)
    fetch + normalization and returns a pydantic model. ``ttl`` overrides the cache's
    default freshness window for this call (e.g. a short window for recency-sensitive news).
    """
    raw = cache.get(key, ttl_override=ttl)
    if raw is not None:
        return model_cls.model_validate_json(raw)
    obj = producer()
    cache.set(key, obj.model_dump_json())
    return obj
