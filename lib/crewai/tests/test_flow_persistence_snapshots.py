"""Tests for portable conversational Flow snapshots.

All tests run offline: no model calls, credentials, console interaction, or
network access. The end-to-end test uses a deterministic route handler.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field
import pytest

from crewai.flow import ConversationConfig, ConversationState, Flow, listen
from crewai.flow.persistence import (
    ConversationSnapshot,
    SQLiteFlowPersistence,
    copy_snapshot,
    persist,
    read_snapshot,
    write_snapshot,
)
from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.snapshots import SNAPSHOT_FORMAT_VERSION


SESSION_ID = "snapshot-test-session"


class MemoryPersistence(FlowPersistence):
    """Minimal backend proving snapshots only need load_state/save_state."""

    persistence_type: str = "MemoryPersistence"
    states: dict[str, dict[str, Any]] = Field(default_factory=dict)
    saved: list[tuple[str, str, dict[str, Any]]] = Field(default_factory=list)
    loads: list[str] = Field(default_factory=list)

    def init_db(self) -> None:
        """No schema is required for an in-memory store."""

    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        """Store one state row and record the call."""
        state = (
            state_data.model_dump(mode="json")
            if isinstance(state_data, BaseModel)
            else dict(state_data)
        )
        self.states[flow_uuid] = state
        self.saved.append((flow_uuid, method_name, state))

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        """Return the latest state row and record the call."""
        self.loads.append(flow_uuid)
        return self.states.get(flow_uuid)


def _state(*messages: tuple[str, str], **extra: Any) -> dict[str, Any]:
    """Build a persisted conversational state blob."""
    return {
        "id": SESSION_ID,
        "messages": [
            {"role": role, "content": content} for role, content in messages
        ],
        **extra,
    }


def _make_bot(store: FlowPersistence) -> type[Flow[ConversationState]]:
    """Build a deterministic conversational Flow bound to one backend."""

    @ConversationConfig(trace_scope="turn")
    @persist(store)
    class SnapshotBot(Flow[ConversationState]):
        conversational = True

        def route_turn(self, context: dict[str, Any]) -> str | None:
            return "echo"

        @listen("echo")
        def handle_echo(self) -> str:
            return f"echo: {self.state.current_user_message}"

    return SnapshotBot


def _roles(flow: Flow[Any]) -> list[tuple[str, str]]:
    """Extract (role, content) pairs from either dict or model messages."""
    return [
        (
            getattr(message, "role", None) or message["role"],
            getattr(message, "content", None) or message["content"],
        )
        for message in flow.state.messages
    ]


class TestReadSnapshot:
    """Reading is strict: one load, no writes, no Flow instantiation."""

    def test_returns_none_for_missing_session(self) -> None:
        backend = MemoryPersistence()

        assert read_snapshot(SESSION_ID, backend) is None
        assert backend.loads == [SESSION_ID]
        assert backend.saved == []

    def test_returns_messages_in_order_without_advancing_the_session(self) -> None:
        backend = MemoryPersistence(states={SESSION_ID: _state(("user", "hi"), ("assistant", "hello"))})

        snapshot = read_snapshot(SESSION_ID, backend)

        assert snapshot is not None
        assert snapshot.session_id == SESSION_ID
        assert snapshot.transcript == [("user", "hi"), ("assistant", "hello")]
        assert backend.loads == [SESSION_ID]
        assert backend.saved == []

    def test_rejects_empty_session_id(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            read_snapshot("", MemoryPersistence())

    def test_reads_custom_backend_without_extra_methods(self) -> None:
        backend = MemoryPersistence(states={SESSION_ID: _state(("user", "hi"))})

        snapshot = read_snapshot(SESSION_ID, backend)

        assert snapshot is not None
        assert [message["role"] for message in snapshot.messages] == ["user"]

    def test_accepts_pydantic_message_models(self) -> None:
        class Message(BaseModel):
            role: str
            content: str

        backend = MemoryPersistence(
            states={
                SESSION_ID: {
                    "id": SESSION_ID,
                    "messages": [Message(role="user", content="hi")],
                }
            }
        )

        snapshot = read_snapshot(SESSION_ID, backend)

        assert snapshot is not None
        assert snapshot.transcript == [("user", "hi")]

    def test_rejects_malformed_messages(self) -> None:
        backend = MemoryPersistence(
            states={SESSION_ID: {"id": SESSION_ID, "messages": "not-a-list"}}
        )

        with pytest.raises(ValueError, match="must be a list"):
            read_snapshot(SESSION_ID, backend)


class TestConversationSnapshot:
    """Snapshot value semantics and JSON portability."""

    def test_is_detached_from_backend_state(self) -> None:
        state = _state(("user", "hi"))

        snapshot = ConversationSnapshot(session_id=SESSION_ID, state=state)
        state["messages"].append({"role": "assistant", "content": "late"})

        assert len(snapshot.messages) == 1

    def test_is_frozen(self) -> None:
        snapshot = ConversationSnapshot(session_id=SESSION_ID, state=_state())

        with pytest.raises(AttributeError):
            snapshot.session_id = "other"  # type: ignore[misc]

    def test_requires_matching_state_id(self) -> None:
        with pytest.raises(ValueError, match="does not match session_id"):
            ConversationSnapshot(
                session_id=SESSION_ID, state={"id": "another-session", "messages": []}
            )

    def test_stamps_session_identity_on_state(self) -> None:
        snapshot = ConversationSnapshot(session_id=SESSION_ID, state={"messages": []})

        assert snapshot.state["id"] == SESSION_ID

    def test_json_round_trip_preserves_nested_state_and_unicode(self) -> None:
        state = _state(
            ("user", "你好，世界"),
            ("assistant", "reply"),
            metadata={"nested": {"count": 3, "tags": ["a", "b"], "empty": None}},
        )

        restored = ConversationSnapshot.from_json(
            ConversationSnapshot(session_id=SESSION_ID, state=state).to_json()
        )

        assert restored.session_id == SESSION_ID
        assert restored.state == state
        assert restored.transcript == [("user", "你好，世界"), ("assistant", "reply")]

    def test_json_round_trip_preserves_pydantic_messages(self) -> None:
        class Message(BaseModel):
            role: str
            content: list[dict[str, Any]]
            tool_calls: list[dict[str, Any]]

        message = Message(
            role="assistant",
            content=[{"type": "text", "text": "你好"}],
            tool_calls=[{"id": "call-1", "function": {"name": "lookup"}}],
        )
        state = {
            "id": SESSION_ID,
            "messages": [{"role": "user", "content": "hi"}, message],
            "metadata": {"count": 2},
        }
        backend = MemoryPersistence(states={SESSION_ID: state})
        snapshot = read_snapshot(SESSION_ID, backend)
        assert snapshot is not None
        original_message = snapshot.state["messages"][1]

        payload = snapshot.to_json()
        restored = ConversationSnapshot.from_json(payload)
        target = MemoryPersistence()
        write_snapshot(restored, target)

        assert restored.messages == snapshot.messages
        assert restored.transcript == snapshot.transcript
        assert restored.state["metadata"] == state["metadata"]
        assert restored.state["messages"][1] == message.model_dump(mode="json")
        assert target.states[SESSION_ID] == restored.state
        assert snapshot.to_json() == payload
        assert snapshot.state["messages"][1] is original_message
        assert isinstance(original_message, Message)
        assert backend.states[SESSION_ID]["messages"][1] is message
        assert backend.saved == []

    def test_json_is_deterministic(self) -> None:
        first = ConversationSnapshot(session_id=SESSION_ID, state={"b": 1, "a": 2})
        second = ConversationSnapshot(session_id=SESSION_ID, state={"a": 2, "b": 1})

        assert first.to_json(indent=None) == second.to_json(indent=None)

    def test_from_json_rejects_unsupported_version(self) -> None:
        payload = json.dumps(
            {
                "format_version": SNAPSHOT_FORMAT_VERSION + 1,
                "session_id": SESSION_ID,
                "state": {},
            }
        )

        with pytest.raises(ValueError, match="unsupported conversation snapshot"):
            ConversationSnapshot.from_json(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "not json",
            "[]",
            "{}",
            json.dumps(
                {
                    "format_version": SNAPSHOT_FORMAT_VERSION,
                    "session_id": "",
                    "state": {},
                }
            ),
            json.dumps(
                {
                    "format_version": SNAPSHOT_FORMAT_VERSION,
                    "session_id": SESSION_ID,
                    "state": [],
                }
            ),
        ],
    )
    def test_from_json_rejects_invalid_payloads(self, payload: str) -> None:
        with pytest.raises(ValueError):
            ConversationSnapshot.from_json(payload)

    def test_transcript_renders_non_text_content(self) -> None:
        snapshot = ConversationSnapshot(
            session_id=SESSION_ID,
            state={
                "id": SESSION_ID,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                    {"role": "assistant", "content": None},
                ],
            },
        )

        assert snapshot.transcript == [
            ("user", '[{"text": "hi", "type": "text"}]'),
            ("assistant", ""),
        ]


class TestWriteSnapshot:
    """Writing appends the newest state row and keeps session identity."""

    def test_writes_detached_state_with_restore_label(self) -> None:
        backend = MemoryPersistence()
        snapshot = ConversationSnapshot(
            session_id=SESSION_ID, state=_state(("user", "hi"))
        )

        write_snapshot(snapshot, backend)

        assert len(backend.saved) == 1
        flow_uuid, method_name, state = backend.saved[0]
        assert flow_uuid == SESSION_ID
        assert method_name == "snapshot_restore"
        assert state["id"] == SESSION_ID
        state["messages"].clear()
        assert len(snapshot.messages) == 1

    def test_accepts_custom_method_name(self) -> None:
        backend = MemoryPersistence()

        write_snapshot(
            ConversationSnapshot(session_id=SESSION_ID, state=_state()),
            backend,
            method_name="manual-import",
        )

        assert backend.saved[0][1] == "manual-import"

    def test_rejects_empty_method_name(self) -> None:
        with pytest.raises(ValueError, match="method_name"):
            write_snapshot(
                ConversationSnapshot(session_id=SESSION_ID, state=_state()),
                MemoryPersistence(),
                method_name="",
            )

    def test_copy_between_backends(self) -> None:
        source = MemoryPersistence(states={SESSION_ID: _state(("user", "hi"))})
        target = MemoryPersistence()

        copied = copy_snapshot(SESSION_ID, source=source, target=target)

        assert copied.session_id == SESSION_ID
        assert target.load_state(SESSION_ID) == source.load_state(SESSION_ID)
        assert source.saved == []

    def test_copy_missing_session_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="no persisted conversation"):
            copy_snapshot(
                SESSION_ID, source=MemoryPersistence(), target=MemoryPersistence()
            )


class TestEndToEndMigration:
    """The runnable documentation example, as a deterministic smoke check."""

    def test_fresh_instance_resumes_after_cross_database_snapshot_copy(
        self, tmp_path: Any
    ) -> None:
        primary = SQLiteFlowPersistence(str(tmp_path / "primary.db"))
        replica = SQLiteFlowPersistence(str(tmp_path / "replica.db"))
        PrimaryBot = _make_bot(primary)
        ReplicaBot = _make_bot(replica)

        PrimaryBot().handle_turn("first question", session_id=SESSION_ID)

        snapshot = read_snapshot(SESSION_ID, primary)
        assert snapshot is not None
        assert snapshot.transcript == [
            ("user", "first question"),
            ("assistant", "echo: first question"),
        ]

        # A portable artifact survives JSON serialization and a new backend.
        restored = ConversationSnapshot.from_json(snapshot.to_json())
        write_snapshot(restored, replica)

        resumed = ReplicaBot()
        resumed.handle_turn("second question", session_id=SESSION_ID)
        assert _roles(resumed) == [
            ("user", "first question"),
            ("assistant", "echo: first question"),
            ("user", "second question"),
            ("assistant", "echo: second question"),
        ]

        separate = ReplicaBot()
        separate.handle_turn("unrelated", session_id="another-session")
        assert _roles(separate) == [
            ("user", "unrelated"),
            ("assistant", "echo: unrelated"),
        ]

    def test_copy_snapshot_restores_without_duplicate_messages(
        self, tmp_path: Any
    ) -> None:
        primary = SQLiteFlowPersistence(str(tmp_path / "primary.db"))
        replica = SQLiteFlowPersistence(str(tmp_path / "replica.db"))
        PrimaryBot = _make_bot(primary)
        ReplicaBot = _make_bot(replica)

        PrimaryBot().handle_turn("one", session_id=SESSION_ID)
        PrimaryBot().handle_turn("two", session_id=SESSION_ID)
        copy_snapshot(SESSION_ID, source=primary, target=replica)

        resumed = ReplicaBot()
        resumed.handle_turn("three", session_id=SESSION_ID)

        assert _roles(resumed) == [
            ("user", "one"),
            ("assistant", "echo: one"),
            ("user", "two"),
            ("assistant", "echo: two"),
            ("user", "three"),
            ("assistant", "echo: three"),
        ]
