"""DiskCache + cached_model behavior."""

from __future__ import annotations

import os
import time
from pathlib import Path

from pydantic import BaseModel

from stock_agent.providers._cache import DiskCache, cached_model, make_key


class _M(BaseModel):
    x: int


def test_make_key_stable_and_order_sensitive() -> None:
    assert make_key("a", 1) == make_key("a", 1)
    assert make_key("a", 1) != make_key("a", 2)
    assert make_key("a", "b") != make_key("b", "a")  # order is significant


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=100)
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    assert DiskCache(tmp_path, ttl_seconds=100).get("absent") is None


def test_cache_ttl_expiry(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=100)
    cache.set("k", "v")
    # Backdate mtime beyond the TTL to force expiry deterministically.
    old = time.time() - 1000
    os.utime(tmp_path / "k.json", (old, old))
    assert cache.get("k") is None


def test_cache_disabled_is_noop(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=100, enabled=False)
    cache.set("k", "v")
    assert cache.get("k") is None


def test_cached_model_produces_once_then_serves_cache(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, ttl_seconds=100)
    calls = {"n": 0}

    def producer() -> _M:
        calls["n"] += 1
        return _M(x=42)

    first = cached_model(cache, "key", _M, producer)
    second = cached_model(cache, "key", _M, producer)
    assert first.x == 42 and second.x == 42
    assert calls["n"] == 1  # second call served from cache
