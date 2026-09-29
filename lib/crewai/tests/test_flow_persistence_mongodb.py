"""Unit tests for MongoDbFlowPersistence.

These never touch a real MongoDB or the network: ``pymongo.MongoClient`` is
patched with a tiny in-memory fake, so they comply with the suite's
``--block-network`` policy and add negligible CI load.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from decimal import Decimal
import os
from pathlib import Path
import sys
from typing import Any
import uuid
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict
import pytest

from crewai.flow import Flow, human_feedback, listen, start
from crewai.flow.async_feedback.types import PendingFeedbackContext
from crewai.flow.flow import FlowState
from crewai.flow.persistence import persist
from crewai.flow.persistence.mongodb import MongoDbFlowPersistence

pytest.importorskip("pymongo")

CONN = "mongodb://localhost:27017"


class _FakeCollection:
    """Minimal in-memory stand-in for a pymongo collection."""

    def __init__(self) -> None:
        self.docs: list[dict[str, Any]] = []
        self.find_one_calls: list[tuple[dict[str, Any], Any]] = []
        self.fail_on_replace = False

    @staticmethod
    def _match(doc: dict[str, Any], flt: dict[str, Any]) -> bool:
        return all(doc.get(k) == v for k, v in flt.items())

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        pass

    def insert_one(self, doc: dict[str, Any], session: Any = None) -> None:
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
        session: Any = None,
    ) -> dict[str, Any] | None:
        row = next((d for d in self.docs if self._match(d, flt)), None)
        if row is None:
            if not upsert:
                return None
            row = dict(flt)
            self.docs.append(row)
        for key, delta in update.get("$inc", {}).items():
            row[key] = row.get(key, 0) + delta
        return dict(row)

    def replace_one(
        self,
        flt: dict[str, Any],
        doc: dict[str, Any],
        upsert: bool = False,
        session: Any = None,
    ) -> None:
        if self.fail_on_replace:
            raise RuntimeError("simulated pending-feedback write failure")
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


class _FakeSession:
    def __init__(self, client: _FakeClient) -> None:
        self.client = client

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def with_transaction(self, callback: Any) -> None:
        self.client.transactions_started += 1
        collections_before = copy.deepcopy(self.client._db.collections)
        try:
            callback(self)
        except Exception:
            self.client._db.collections = collections_before
            raise


class _FakeClient:
    def __init__(self, conn: str) -> None:
        self.conn = conn
        self.db_names: list[str] = []
        self._db = _FakeDatabase()
        self.transactions_started = 0

    def __getitem__(self, name: str) -> _FakeDatabase:
        self.db_names.append(name)
        return self._db

    def start_session(self) -> _FakeSession:
        return _FakeSession(self)


def _patch_client(
    monkeypatch: pytest.MonkeyPatch, *, reuse_client: bool = False
) -> dict[str, Any]:
    """Patch ``pymongo.MongoClient`` to build in-memory fakes.

    Returns a dict capturing the connection string and the created fake client
    so tests can assert on call-time resolution and stored documents.
    """
    import pymongo

    created: dict[str, Any] = {}
    shared_client: _FakeClient | None = None

    def factory(conn: str, *args: Any, **kwargs: Any) -> _FakeClient:
        nonlocal shared_client
        if reuse_client:
            if shared_client is None:
                shared_client = _FakeClient(conn)
            client = shared_client
        else:
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


def test_persistences_sharing_a_state_collection_share_its_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate persistence instances keep one ordering sequence per collection."""
    created = _patch_client(monkeypatch, reuse_client=True)
    first = MongoDbFlowPersistence(CONN)
    second = MongoDbFlowPersistence(CONN)

    first.save_state("flow-1", "first", {"counter": 1})
    second.save_state("flow-1", "second", {"counter": 2})

    states = created["client"]._db["flow_states"]
    assert [doc["seq"] for doc in states.docs] == [1, 2]
    assert first.load_state("flow-1") == {"counter": 2}


def test_different_state_collections_use_independent_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each configured state collection has an independent ordering sequence."""
    created = _patch_client(monkeypatch, reuse_client=True)
    default_states = MongoDbFlowPersistence(CONN)
    billing_states = MongoDbFlowPersistence(CONN, states_collection="billing_states")

    default_states.save_state("flow-1", "default", {"counter": 1})
    billing_states.save_state("flow-1", "billing", {"counter": 1})

    database = created["client"]._db
    assert database["flow_states"].docs[0]["seq"] == 1
    assert database["billing_states"].docs[0]["seq"] == 1


def test_save_state_assigns_sequence_and_inserts_in_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)

    persistence.save_state("flow-1", "step", {"counter": 1})

    assert created["client"].transactions_started == 1


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


def test_basemodel_state_falls_back_to_python_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch)

    class NonJsonValue:
        def __str__(self) -> str:
            return "non-json value"

    class State(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        value: NonJsonValue

    persistence = MongoDbFlowPersistence(CONN)

    persistence.save_state("flow-1", "s", State(value=NonJsonValue()))

    assert persistence.load_state("flow-1") == {"value": "non-json value"}


def test_dict_state_serializes_non_json_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch)

    class NestedState(BaseModel):
        when: datetime

    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    persistence = MongoDbFlowPersistence(CONN)

    persistence.save_state(
        "flow-1",
        "s",
        {
            "when": when,
            "tags": {"a", "b"},
            "items": (1, 2),
            "nested": NestedState(when=when),
        },
    )

    loaded = persistence.load_state("flow-1")

    assert loaded is not None
    assert loaded["when"] == "2026-01-02T03:04:05+00:00"
    assert set(loaded["tags"]) == {"a", "b"}
    assert loaded["items"] == [1, 2]
    assert loaded["nested"] == {"when": "2026-01-02T03:04:05Z"}


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
    created = _patch_client(monkeypatch)
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
    assert created["client"].transactions_started == 1


def test_pending_feedback_replaces_existing_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving feedback twice matches SQLite's INSERT OR REPLACE behavior."""
    _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)
    first_context = PendingFeedbackContext(
        flow_id="flow-1",
        flow_class="MyFlow",
        method_name="first_review",
        method_output="first draft",
        message="First question?",
    )
    second_context = PendingFeedbackContext(
        flow_id="flow-1",
        flow_class="MyFlow",
        method_name="second_review",
        method_output="second draft",
        message="Second question?",
    )

    persistence.save_pending_feedback("flow-1", first_context, {"counter": 1})
    persistence.save_pending_feedback("flow-1", second_context, {"counter": 2})

    loaded = persistence.load_pending_feedback("flow-1")
    assert loaded is not None
    state, context = loaded
    assert state == {"counter": 2}
    assert context.method_name == "second_review"
    assert context.message == "Second question?"


def test_pending_feedback_write_failure_rolls_back_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)
    context = PendingFeedbackContext(
        flow_id="flow-1",
        flow_class="MyFlow",
        method_name="review",
        method_output={"draft": "hi"},
        message="Approve?",
    )
    persistence._db_ready()[persistence.pending_collection].fail_on_replace = True

    with pytest.raises(RuntimeError, match="simulated pending-feedback"):
        persistence.save_pending_feedback("flow-1", context, {"counter": 3})

    assert persistence.load_state("flow-1") is None
    assert persistence.load_pending_feedback("flow-1") is None


def test_persisted_flow_restores_latest_mongodb_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)

    class State(FlowState):
        counter: int = 0

    class PersistedFlow(Flow[State]):
        @start()
        @persist(persistence)
        def first_step(self) -> None:
            self.state.counter += 1

        @listen("first_step")
        @persist(persistence)
        def second_step(self) -> None:
            self.state.counter += 1

    first_run = PersistedFlow(persistence=persistence)
    first_run.kickoff()
    flow_id = first_run.state.id

    restored_run = PersistedFlow(persistence=persistence)
    restored_run.kickoff(inputs={"id": flow_id})

    assert first_run.state.counter == 2
    assert restored_run.state.counter == 4
    assert persistence.load_state(flow_id) == {"id": flow_id, "counter": 4}


def test_mongodb_fork_restores_source_state_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mongo persistence follows SQLite's restore_from_state_id fork semantics."""
    _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)

    class State(FlowState):
        counter: int = 0

    class ForkableFlow(Flow[State]):
        @start()
        @persist(persistence)
        def step(self) -> None:
            self.state.counter += 1

    source = ForkableFlow(persistence=persistence)
    source.kickoff()
    source_id = source.state.id

    resumed_source = ForkableFlow(persistence=persistence)
    resumed_source.kickoff(inputs={"id": source_id})
    assert persistence.load_state(source_id) == {"id": source_id, "counter": 2}

    fork = ForkableFlow(persistence=persistence)
    fork.kickoff(restore_from_state_id=source_id)

    assert fork.state.id != source_id
    assert fork.state.counter == 3
    assert persistence.load_state(source_id) == {"id": source_id, "counter": 2}
    assert persistence.load_state(fork.state.id) == {
        "id": fork.state.id,
        "counter": 3,
    }


def test_mongodb_from_pending_resumes_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)

    class ReviewFlow(Flow[dict[str, Any]]):
        @start()
        @human_feedback(message="Review this:")
        def generate(self) -> str:
            return "generated content"

        @listen(generate)
        def process(self, feedback_result: Any) -> str:
            return f"Processed: {feedback_result.feedback}"

    context = PendingFeedbackContext(
        flow_id="resume-flow-1",
        flow_class="test.ReviewFlow",
        method_name="generate",
        method_output="generated content",
        message="Review this:",
    )
    persistence.save_pending_feedback(
        "resume-flow-1", context, {"id": "resume-flow-1"}
    )

    flow = ReviewFlow.from_pending("resume-flow-1", persistence)
    with patch("crewai.flow.runtime.crewai_event_bus.emit"):
        flow.resume("looks good!")

    assert flow.last_human_feedback is not None
    assert flow.last_human_feedback.feedback == "looks good!"
    assert persistence.load_pending_feedback("resume-flow-1") is None


def test_persisted_flow_serializes_complex_mongodb_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_client(monkeypatch)
    persistence = MongoDbFlowPersistence(CONN)
    now = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    flow_user_id = uuid.uuid4()

    class ComplexState(FlowState):
        created_at: datetime = now
        user_id: uuid.UUID = flow_user_id
        tags: set[str] = {"alpha", "beta"}
        price: Decimal = Decimal("19.99")
        file_path: Path = Path("/tmp/data.txt")

    class ComplexFlow(Flow[ComplexState]):
        @start()
        @persist(persistence)
        def step(self) -> None:
            pass

    flow = ComplexFlow(persistence=persistence)
    flow.kickoff()

    saved = persistence.load_state(flow.state.id)
    assert saved is not None
    assert datetime.fromisoformat(saved["created_at"].replace("Z", "+00:00")) == now
    assert saved["user_id"] == str(flow_user_id)
    assert set(saved["tags"]) == {"alpha", "beta"}
    assert saved["price"] == "19.99"
    assert saved["file_path"] == "/tmp/data.txt"


@pytest.mark.skipif(
    not os.getenv("MONGODB_TEST_CONNECTION_STRING"),
    reason="requires a MongoDB replica-set URI in MONGODB_TEST_CONNECTION_STRING",
)
def test_pending_feedback_transaction_rolls_back_in_mongodb() -> None:
    """Verify a real MongoDB transaction rolls back a failed feedback upsert.

    Run this test against a replica set with network blocking disabled.
    """
    from pymongo import MongoClient
    from pymongo.errors import OperationFailure

    connection_string = os.environ["MONGODB_TEST_CONNECTION_STRING"]
    database_name = f"crewai_persistence_test_{uuid.uuid4().hex}"
    client: Any = MongoClient(connection_string, serverSelectionTimeoutMS=5_000)
    database = client[database_name]
    database.create_collection(
        "pending_feedback",
        validator={"$jsonSchema": {"bsonType": "object", "required": ["blocked"]}},
    )
    persistence = MongoDbFlowPersistence(connection_string, database_name=database_name)
    context = PendingFeedbackContext(
        flow_id="flow-1",
        flow_class="MyFlow",
        method_name="review",
        method_output={"draft": "hi"},
        message="Approve?",
    )

    try:
        with pytest.raises(OperationFailure):
            persistence.save_pending_feedback("flow-1", context, {"counter": 3})

        assert persistence.load_state("flow-1") is None
        assert persistence.load_pending_feedback("flow-1") is None
    finally:
        client.drop_database(database_name)
        client.close()


def test_missing_pymongo_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate pymongo not being installed so the lazy import fails.
    monkeypatch.setitem(sys.modules, "pymongo", None)
    persistence = MongoDbFlowPersistence(CONN)

    with pytest.raises(ImportError, match=r"crewai\[mongodb\]"):
        persistence.load_state("flow-1")
