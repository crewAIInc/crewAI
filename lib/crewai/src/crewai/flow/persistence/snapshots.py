"""Portable snapshots of persisted conversational Flow sessions.

``FlowPersistence`` stores the latest state blob for each flow UUID. Reading
that blob normally means constructing a Flow and running a turn, which
*mutates* the conversation. This module exposes the stored state as an
immutable, JSON-portable :class:`ConversationSnapshot` so applications can
inspect, back up, seed, and migrate a conversation without advancing it.

Snapshots are backend-agnostic: they only use
``FlowPersistence.load_state()`` and ``FlowPersistence.save_state()``. Custom
backends need no new methods. Unlike ``CheckpointConfig`` checkpoints, a
snapshot contains only the persisted conversation state — never completed
methods, graph progress, or other execution resources.

Example:
    ```python
    from crewai.flow.persistence import SQLiteFlowPersistence
    from crewai.flow.persistence.snapshots import (
        copy_snapshot,
        read_snapshot,
    )

    primary = SQLiteFlowPersistence("primary.db")
    replica = SQLiteFlowPersistence("replica.db")

    snapshot = read_snapshot("customer-4471", primary)
    if snapshot is not None:
        print(snapshot.transcript)
        with open("customer-4471.json", "w", encoding="utf-8") as handle:
            handle.write(snapshot.to_json())

    copy_snapshot("customer-4471", source=primary, target=replica)
    ```
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import TYPE_CHECKING, Any, Final, cast

from pydantic import BaseModel


if TYPE_CHECKING:
    from crewai.flow.persistence.base import FlowPersistence
    from crewai.utilities.types import LLMMessage


SNAPSHOT_FORMAT_VERSION: Final[int] = 1
DEFAULT_RESTORE_METHOD_NAME: Final[str] = "snapshot_restore"


def _render_content(content: Any) -> str:
    """Render an LLM message payload as displayable text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    """An immutable copy of one persisted conversational Flow session.

    Attributes:
        session_id: Flow UUID the snapshot belongs to. Always written back as
            ``state["id"]`` so a restored Flow keeps the same session identity.
        state: Deep copy of the most recent persisted state blob.
    """

    session_id: str
    state: dict[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        """Validate identity and detach the state from the backend's object."""
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.state, dict):
            raise ValueError("snapshot state must be a dict")

        state = deepcopy(self.state)
        if "id" in state and state["id"] != self.session_id:
            raise ValueError(
                "snapshot state 'id' does not match session_id: "
                f"{state['id']!r} != {self.session_id!r}"
            )
        state["id"] = self.session_id
        object.__setattr__(self, "state", state)
        # Fail early on malformed persisted messages rather than at read time.
        _ = self.messages

    @property
    def messages(self) -> tuple[LLMMessage, ...]:
        """Conversation messages in persisted order, as detached plain dicts.

        Persisted rows normally hold JSON dicts, but a custom backend may
        return Pydantic models. Both shapes are normalized here.

        Raises:
            ValueError: If ``state["messages"]`` is not a list of mappings.
        """
        raw = self.state.get("messages")
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise ValueError("persisted 'messages' must be a list or null")

        messages: list[LLMMessage] = []
        for entry in raw:
            if isinstance(entry, BaseModel):
                messages.append(
                    cast("LLMMessage", deepcopy(entry.model_dump(mode="json")))
                )
            elif isinstance(entry, Mapping):
                messages.append(cast("LLMMessage", deepcopy(dict(entry))))
            else:
                raise ValueError("persisted 'messages' entries must be mappings")
        return tuple(messages)

    @property
    def transcript(self) -> list[tuple[str, str]]:
        """``(role, content)`` pairs in persisted order."""
        return [
            (
                str(message.get("role", "unknown")),
                _render_content(message.get("content")),
            )
            for message in self.messages
        ]

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to a deterministic, versioned JSON artifact.

        Args:
            indent: Indentation passed to ``json.dumps``. Use ``None`` for a
                single-line artifact.

        Returns:
            The snapshot as JSON with sorted keys and unescaped Unicode.
        """
        state = dict(self.state)
        if state.get("messages") is not None:
            # Use the same structured normalization as reads without mutating
            # the snapshot or the custom backend's Pydantic message objects.
            state["messages"] = list(self.messages)
        document: dict[str, Any] = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "session_id": self.session_id,
            "state": state,
        }
        return json.dumps(
            document,
            sort_keys=True,
            ensure_ascii=False,
            indent=indent,
            default=str,
        )

    @classmethod
    def from_json(cls, payload: str) -> ConversationSnapshot:
        """Rebuild a snapshot from :meth:`to_json` output.

        Args:
            payload: JSON produced by :meth:`to_json`.

        Returns:
            A validated snapshot, ready for :func:`write_snapshot`.

        Raises:
            ValueError: If the payload is malformed, uses an unsupported
                format version, or has an inconsistent session identity.
        """
        try:
            document = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"invalid conversation snapshot JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError("conversation snapshot JSON must be an object")

        version = document.get("format_version")
        if version != SNAPSHOT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported conversation snapshot format_version {version!r}; "
                f"expected {SNAPSHOT_FORMAT_VERSION}"
            )
        session_id = document.get("session_id")
        state = document.get("state")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(
                "conversation snapshot 'session_id' must be a non-empty string"
            )
        if not isinstance(state, dict):
            raise ValueError("conversation snapshot 'state' must be an object")

        return cls(session_id=session_id, state=state)


def read_snapshot(
    session_id: str, persistence: FlowPersistence
) -> ConversationSnapshot | None:
    """Read a persisted conversation without running a turn.

    This is a strict read: it calls ``load_state()`` only and never creates,
    writes, or advances a session. Constructing a backend may itself perform
    setup (for example, ``SQLiteFlowPersistence`` creates its schema).

    Args:
        session_id: Flow UUID / conversational session id to read.
        persistence: Backend holding the persisted state.

    Returns:
        The snapshot, or ``None`` when the session has no persisted state.

    Raises:
        ValueError: If ``session_id`` is empty or the stored state is invalid.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty string")

    state = persistence.load_state(session_id)
    if state is None:
        return None
    return ConversationSnapshot(session_id=session_id, state=state)


def write_snapshot(
    snapshot: ConversationSnapshot,
    persistence: FlowPersistence,
    *,
    method_name: str = DEFAULT_RESTORE_METHOD_NAME,
) -> None:
    """Write a snapshot as the latest state for its session.

    The snapshot becomes the newest row, which is what ``load_state()`` and
    ``@persist`` restore. Existing rows are left intact, so backend history
    and audit trails are preserved.

    Args:
        snapshot: Snapshot to restore.
        persistence: Destination backend; may differ from the source backend.
        method_name: Label recorded alongside the restored row.

    Raises:
        ValueError: If ``method_name`` is empty.
    """
    if not isinstance(method_name, str) or not method_name:
        raise ValueError("method_name must be a non-empty string")

    persistence.save_state(
        flow_uuid=snapshot.session_id,
        method_name=method_name,
        state_data=deepcopy(snapshot.state),
    )


def copy_snapshot(
    session_id: str,
    *,
    source: FlowPersistence,
    target: FlowPersistence,
    method_name: str = DEFAULT_RESTORE_METHOD_NAME,
) -> ConversationSnapshot:
    """Copy one conversation between persistence backends.

    Args:
        session_id: Session to copy.
        source: Backend to read from.
        target: Backend to write to.
        method_name: Label recorded alongside the restored row.

    Returns:
        The snapshot that was copied.

    Raises:
        KeyError: If the source backend has no state for ``session_id``.
    """
    snapshot = read_snapshot(session_id, source)
    if snapshot is None:
        raise KeyError(f"no persisted conversation found for session {session_id!r}")
    write_snapshot(snapshot, target, method_name=method_name)
    return snapshot


__all__ = [
    "DEFAULT_RESTORE_METHOD_NAME",
    "SNAPSHOT_FORMAT_VERSION",
    "ConversationSnapshot",
    "copy_snapshot",
    "read_snapshot",
    "write_snapshot",
]
