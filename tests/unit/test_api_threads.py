"""Contract tests for the P2.5b thread-CRUD endpoints (``/threads``).

Deterministic (no network/LLM): the ``ChatStore`` dependency is overridden to a ``tmp_path`` store
so tests never touch the real ``outputs/chat_history`` tree. Asserts the create → list → get →
delete roundtrip the React sidebar relies on, title derivation, ``created_at`` preservation on
update (upsert in place, no duplicate), most-recent-first ordering, and 404 on a missing id.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stock_agent.api.app import create_app
from stock_agent.api.deps import chat_store_dep
from stock_agent.chat.history import ChatStore


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = create_app()
    store = ChatStore(tmp_path / "threads", retention_days=30)
    app.dependency_overrides[chat_store_dep] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def _msgs(text: str) -> list[dict[str, object]]:
    """A minimal two-message display transcript (title derives from the first user message)."""
    return [{"role": "user", "content": text}, {"role": "assistant", "content": "ok"}]


def test_create_then_list_and_get(client: TestClient) -> None:
    # Empty id -> server mints one; empty title -> derived from the first user message.
    r = client.post("/threads", json={"display_messages": _msgs("What is NVDA volatility?")})
    assert r.status_code == 200
    body = r.json()
    assert body["id"]  # a fresh id was minted
    assert body["title"] == "What is NVDA volatility?"
    assert body["created_at"] == body["updated_at"]  # equal at creation
    tid = body["id"]

    listed = client.get("/threads").json()
    assert [m["id"] for m in listed] == [tid]
    assert listed[0]["title"] == "What is NVDA volatility?"

    got = client.get(f"/threads/{tid}").json()
    assert got["display_messages"] == _msgs("What is NVDA volatility?")
    assert got["agent_history"] == []  # display-level persistence; no resumable history yet


def test_update_preserves_created_at_and_upserts_in_place(client: TestClient) -> None:
    first = client.post("/threads", json={"display_messages": _msgs("hi")}).json()
    tid, created = first["id"], first["created_at"]

    # Re-save under the same id with more messages + an explicit title.
    upd = client.post(
        "/threads",
        json={"id": tid, "title": "Renamed", "display_messages": _msgs("hi there")},
    ).json()
    assert upd["id"] == tid
    assert upd["created_at"] == created  # creation time preserved across the update
    assert upd["updated_at"] >= created  # ISO strings sort by time; stamped now
    assert upd["title"] == "Renamed"  # explicit title wins over derivation

    listed = client.get("/threads").json()
    assert len(listed) == 1  # updated in place, not duplicated
    assert client.get(f"/threads/{tid}").json()["display_messages"] == _msgs("hi there")


def test_delete_is_idempotent_and_missing_get_is_404(client: TestClient) -> None:
    tid = client.post("/threads", json={"display_messages": _msgs("bye")}).json()["id"]
    assert client.delete(f"/threads/{tid}").status_code == 204
    assert client.get("/threads").json() == []
    # Deleting again is a no-op (still 204); getting a missing thread is 404 (never 500).
    assert client.delete(f"/threads/{tid}").status_code == 204
    assert client.get(f"/threads/{tid}").status_code == 404


def test_list_is_most_recent_first(client: TestClient) -> None:
    a = client.post("/threads", json={"display_messages": _msgs("first")}).json()["id"]
    b = client.post("/threads", json={"display_messages": _msgs("second")}).json()["id"]
    ids = [m["id"] for m in client.get("/threads").json()]
    assert set(ids) == {a, b}
    assert ids[0] == b  # b created after a -> sorts first (updated_at descending)


def test_empty_message_thread_gets_fallback_title(client: TestClient) -> None:
    # No user message -> derive_title falls back to "New chat" (no crash on empty transcript).
    body = client.post("/threads", json={"display_messages": []}).json()
    assert body["title"] == "New chat"
