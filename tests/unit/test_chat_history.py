"""Chat-thread store: save/load round-trip, listing, pruning, title, serialization."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stock_agent.chat.history import (
    ChatStore,
    ChatThread,
    derive_title,
    new_thread_id,
    now_iso,
    to_jsonable,
)


def _thread(tid: str = "t1", *, updated: str | None = None) -> ChatThread:
    ts = updated or now_iso()
    return ChatThread(
        id=tid,
        title="DELL news + forecast",
        created_at=ts,
        updated_at=ts,
        display_messages=[
            {"role": "user", "content": "summarize DELL news"},
            {
                "role": "assistant",
                "content": "Here are the themes.",
                "charts": [
                    {
                        "title": "News insights by category",
                        "kind": "bar",
                        "x": "category",
                        "y": "count",
                        "caption": "",
                        "color": None,
                        "x_sort": ["Bullish", "Bearish"],
                        "y_is_percent": False,
                        "data": {"category": ["Bullish", "Bearish"], "count": [2, 1]},
                    }
                ],
            },
        ],
        agent_history=[{"role": "user", "content": "summarize DELL news"}],
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    t = _thread()
    store.save(t)
    loaded = store.load("t1")
    assert loaded.id == "t1"
    assert loaded.title == "DELL news + forecast"
    assert loaded.display_messages[0]["content"] == "summarize DELL news"
    # The chart dict survives intact (UI rebuilds a ChartSpec from it).
    chart = loaded.display_messages[1]["charts"][0]
    assert chart["data"] == {"category": ["Bullish", "Bearish"], "count": [2, 1]}
    assert loaded.agent_history == [{"role": "user", "content": "summarize DELL news"}]


def test_list_threads_sorted_by_recency(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    old = now_iso()
    newer = datetime.fromisoformat(old) + timedelta(hours=1)
    store.save(_thread("old", updated=old))
    store.save(_thread("new", updated=newer.isoformat()))
    ids = [m.id for m in store.list_threads()]
    assert ids == ["new", "old"]  # most-recently-updated first


def test_prune_removes_expired_only(tmp_path: Path) -> None:
    store = ChatStore(tmp_path, retention_days=30)
    fresh = now_iso()
    stale = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    store.save(_thread("fresh", updated=fresh))
    store.save(_thread("stale", updated=stale))
    removed = store.prune()
    assert removed == 1
    assert [m.id for m in store.list_threads()] == ["fresh"]


def test_delete(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    store.save(_thread("t1"))
    store.delete("t1")
    assert store.list_threads() == []
    store.delete("t1")  # idempotent (missing_ok)


def test_list_skips_corrupt_files(tmp_path: Path) -> None:
    store = ChatStore(tmp_path)
    store.save(_thread("good"))
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert [m.id for m in store.list_threads()] == ["good"]


def test_derive_title_from_first_user_message() -> None:
    msgs = [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "  Forecast   MU  "},
    ]
    assert derive_title(msgs) == "Forecast MU"
    assert derive_title([]) == "New chat"
    long = "x" * 100
    assert derive_title([{"role": "user", "content": long}]).endswith("…")


def test_to_jsonable_normalizes_sdk_blocks() -> None:
    """agent_history may hold Anthropic SDK content blocks (model_dump) — normalize them."""

    class _Block:
        def model_dump(self) -> dict[str, Any]:
            return {"type": "text", "text": "hi"}

    out = to_jsonable({"role": "assistant", "content": [_Block()]})
    assert out == {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}


def test_new_thread_id_is_unique() -> None:
    assert new_thread_id() != new_thread_id()
