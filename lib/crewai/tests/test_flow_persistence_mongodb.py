"""Unit tests for MongoDbFlowPersistence.

These never touch a real MongoDB or the network: ``pymongo.MongoClient`` is
patched with a tiny in-memory fake, so they comply with the suite's
``--block-network`` policy and add negligible CI load.
"""

from __future__ import annotations

from datetime import datetime, timezone
import sys
from typing import Any

from pydantic import BaseModel
import pytest

from crewai.flow.async_feedback.types import PendingFeedbackContext
from crewai.flow.persistence.mongodb import MongoDbFlowPersistence

pytest.importorskip("pymongo")

CONN = "mongodb://localhost:27017"


class _FakeCollection:
    """Minimal in-memory stand-in for a pymongo collection."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self.find_one_calls: list[tuple[dict[str, Any], Any]] = []

    @staticmethod
    def _match(doc: dict[str, Any], flt: dict[str, Any]) -> bool:
        return all(doc.get(k) == v for k, v in flt.items())

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        pass

    def insert_one(self, doc: dict[str, Any]) -> None:
        self.docs.append(dict(doc))

    def find_one(
        self, flt: dict[str, Any], sort: list[tuple[str, int]] | None = None
    ) -> dict[str, Any] | None:
        self.find_one_calls.append((dict(flt), sort))
        rows = [d for d in self.docs if self._match(d, flt)]
        if sort:
            for key, direction in reversed(sort):
                rows.sort(key=lambda d: d.get(key, 0), reverse=direction < 0)
        return dict(rows[0]) if rows else None

    def find_one_and_update(
        self,
        flt: dict[str, Any],
        update: dict[str, Any],
        upsert: bool = False,
        return_document: Any = None,
    ) -> dict[str, Any] | None:
        row = next((d for d in self.docs if self._match(d, flt)), None)
        if row is None and upsert:
            row = dict(flt)
            self.docs.append(row)
        for key, delta in update.get("$inc", {}).items():
            row[key] = row.get(key, 0) + delta
        return dict(row)

    def replace_one(
        self, flt: dict[str, Any], doc: dict[str, Any], upsert: bool = False
    ) -> None:
        for i, existing in enumerate(self.docs):
            if self._match(existing, flt):
                self.docs[i] = dict(doc)
                return
        if upsert:
            self.docs.append(dict(doc))

    def delete_one(self, flt: dict[str, Any]) -> None:
        for i, existing in enumerate(self.docs):
            if self._match(existing, flt):
                del self.docs[i]
                return


class _FakeDatabase:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def __getitem__(self, name: str) -> _FakeCollection:
        return self.collections.setdefault(name, _FakeCollection())


class _FakeClient:
    def __init__(self, conn: str) -> None:
        self.conn = conn
        self.db_names: list[str] = []
        self._db = _FakeDatabase()

    def __getitem__(self, name: str) -> _FakeDatabase:
        self.db_names.append(name)
        return self._db


def _patch_client(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch ``pymongo.MongoClient`` to build in-memory fakes.

    Returns a dict capturing the connection string and the created fake client
    so tests can assert on call-time resolution and stored documents.
    """
    import pymongo

    created: dict[str, Any] = {}

    def factory(conn: str, *args: Any, **kwargs: Any) -> _FakeClient:
        client = _FakeClient(conn)
        created["conn"] = conn
        created["client"] = client
        return client

    monkeypatch.setattr(pymongo, "MongoClient", factory)
    return created


def test_save_and_load_roundtrip_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)

    persistence.save_state("flow-1", "step", {"counter": 1})

    assert persistence.load_state("flow-1") == {"counter": 1}


def test_load_returns_latest_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)

    persistence.save_state("flow-1", "s1", {"counter": 1})
    persistence.save_state("flow-1", "s2", {"counter": 2})

    assert persistence.load_state("flow-1") == {"counter": 2}


def test_save_state_tags_incrementing_seq(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)

    persistence.save_state("flow-1", "s1", {"n": 1})
    persistence.save_state("flow-1", "s2", {"n": 2})

    states = created["client"]._db["flow_states"]
    assert [doc["seq"] for doc in states.docs] == [1, 2]

    persistence.load_state("flow-1")
    last_filter, last_sort = states.find_one_calls[-1]
    assert last_filter == {"flow_uuid": "flow-1"}
    assert last_sort == [("seq", -1)]


def test_basemodel_state_serialized_as_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch)

    class State(BaseModel):
        when: datetime

    persistence = MongoDbFlowPersistence(CONN)
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    persistence.save_state("flow-1", "s", State(when=when))

    loaded = persistence.load_state("flow-1")
    assert loaded is not None
    # mode="json" serializes datetime to an ISO string; plain model_dump would
    # keep a datetime object and blow up json.dumps.
    assert isinstance(loaded["when"], str)
    assert loaded["when"].startswith("2026-01-02T03:04:05")


def test_missing_connection_string_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch)
    monkeypatch.delenv("MONGODB_CONNECTION_STRING", raising=False)
    persistence = MongoDbFlowPersistence()

    with pytest.raises(ValueError, match="MONGODB_CONNECTION_STRING"):
        persistence.load_state("flow-1")


def test_env_resolved_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    created = _patch_client(monkeypatch)
    # No connection string at construction time.
    persistence = MongoDbFlowPersistence()

    # Env set only after construction; it must be read on first use.
    monkeypatch.setenv("MONGODB_CONNECTION_STRING", CONN)
    monkeypatch.setenv("MONGODB_DATABASE", "custom_db")

    persistence.save_state("flow-1", "s", {"n": 1})

    assert created["conn"] == CONN
    assert "custom_db" in created["client"].db_names


def test_pending_feedback_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)
    context = PendingFeedbackContext(
        flow_id="flow-1",
        flow_class="MyFlow",
        method_name="review",
        method_output={"draft": "hi"},
        message="Approve?",
    )

    persistence.save_pending_feedback("flow-1", context, {"counter": 3})

    loaded = persistence.load_pending_feedback("flow-1")
    assert loaded is not None
    state, loaded_context = loaded
    assert state == {"counter": 3}
    assert loaded_context.method_name == "review"
    assert loaded_context.flow_id == "flow-1"

    persistence.clear_pending_feedback("flow-1")
    assert persistence.load_pending_feedback("flow-1") is None


def test_missing_pymongo_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate pymongo not being installed so the lazy import fails.
    monkeypatch.setitem(sys.modules, "pymongo", None)
    persistence = MongoDbFlowPersistence(CONN)

    with pytest.raises(ImportError, match=r"crewai\[mongodb\]"):
        persistence.load_state("flow-1")
