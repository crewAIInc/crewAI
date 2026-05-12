"""ConversationalProvider protocol and basic implementations."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from crewai.new_agent.models import AgentStatus, Message, ProvenanceEntry

logger = logging.getLogger(__name__)


@runtime_checkable
class ConversationStorage(Protocol):
    """Pluggable persistence for conversation history and provenance.

    OSS ships SQLiteConversationStorage. Enterprise can replace with
    Postgres, DynamoDB, etc.
    """

    def load_messages(self) -> list[Message]: ...
    def save_messages(self, messages: list[Message]) -> None: ...
    def clear_messages(self) -> None: ...
    def load_provenance(self) -> list[ProvenanceEntry]: ...
    def save_provenance(self, entries: list[ProvenanceEntry]) -> None: ...


class SQLiteConversationStorage:
    """Thread-safe SQLite WAL storage for conversations and provenance."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provenance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_json TEXT NOT NULL
                )
            """)

    def load_messages(self) -> list[Message]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT data_json FROM messages ORDER BY id"
                ).fetchall()
            return [Message.model_validate(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug(f"Failed to load messages: {e}")
            return []

    def save_messages(self, messages: list[Message]) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM messages")
                conn.executemany(
                    "INSERT INTO messages (data_json) VALUES (?)",
                    [(json.dumps(m.model_dump(mode="json"), default=str),) for m in messages],
                )
        except Exception as e:
            logger.debug(f"Failed to save messages: {e}")

    def clear_messages(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM messages")
        except Exception as e:
            logger.debug(f"Failed to clear messages: {e}")

    def load_provenance(self) -> list[ProvenanceEntry]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT data_json FROM provenance ORDER BY id"
                ).fetchall()
            return [ProvenanceEntry.model_validate(json.loads(r[0])) for r in rows]
        except Exception as e:
            logger.debug(f"Failed to load provenance: {e}")
            return []

    def save_provenance(self, entries: list[ProvenanceEntry]) -> None:
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM provenance")
                conn.executemany(
                    "INSERT INTO provenance (data_json) VALUES (?)",
                    [(json.dumps(e.model_dump(mode="json"), default=str),) for e in entries],
                )
        except Exception as e:
            logger.debug(f"Failed to save provenance: {e}")


@runtime_checkable
class ConversationalProvider(Protocol):
    """Pluggable transport for agent conversations.

    OSS provides CLIProvider (TUI). Enterprise provides
    SlackProvider, TeamsProvider, WebProvider, etc.
    """

    async def send_message(self, message: Message) -> None: ...
    async def receive_message(self) -> Message: ...
    async def send_status(self, status: AgentStatus) -> None: ...
    def get_history(self) -> list[Message]: ...
    def save_history(self, messages: list[Message]) -> None: ...
    def reset_history(self) -> None: ...
    def save_provenance(self, entries: list[ProvenanceEntry]) -> None: ...
    def load_provenance(self) -> list[ProvenanceEntry]: ...

    def get_scope(self) -> dict[str, str]:
        """Return scope context for multi-tenant memory isolation.

        Enterprise providers override this to convey conversation scope
        (e.g., Slack channel ID, Teams thread, user DM). The executor
        passes this to memory operations so memories are scoped correctly.

        Returns a dict with provider-defined keys. Common keys:
          - "channel_id": platform channel/thread identifier
          - "user_id": platform user identifier
          - "team_id": workspace/org identifier
        """
        ...


class DirectProvider:
    """In-process provider for programmatic use (no TUI, no stdin).

    Conversations happen via message()/amessage() calls directly.
    History is kept in-memory.
    """

    def __init__(self) -> None:
        self._history: list[Message] = []
        self._provenance: list[ProvenanceEntry] = []
        self._pending_status: AgentStatus | None = None

    async def send_message(self, message: Message) -> None:
        self._history.append(message)

    async def receive_message(self) -> Message:
        raise NotImplementedError(
            "DirectProvider does not support interactive receive. "
            "Use agent.message() instead."
        )

    async def send_status(self, status: AgentStatus) -> None:
        self._pending_status = status

    def get_history(self) -> list[Message]:
        return list(self._history)

    def save_history(self, messages: list[Message]) -> None:
        self._history = list(messages)

    def reset_history(self) -> None:
        self._history.clear()

    def save_provenance(self, entries: list[ProvenanceEntry]) -> None:
        """Persist provenance entries in memory."""
        self._provenance = list(entries)

    def load_provenance(self) -> list[ProvenanceEntry]:
        """Load provenance entries from memory."""
        return list(self._provenance)

    def get_scope(self) -> dict[str, str]:
        return {}
