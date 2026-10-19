"""SQLite-based implementation of flow state persistence."""

from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any

from crewai_core.lock_store import lock as store_lock
from crewai_core.paths import db_storage_path
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.flow.persistence.base import (
    DEFAULT_HISTORY_LIMIT,
    MAX_HISTORY_LIMIT,
    FlowPersistence,
    FlowStateHistoryRecord,
)


if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import PendingFeedbackContext


_SQLITE_BUSY_TIMEOUT_SECONDS = 30.0
_SCHEMA_RETRY_DELAY_SECONDS = 0.05
_SQLITE_MAX_INTEGER = 2**63 - 1


def _json_default(obj: Any) -> Any:
    """Fallback serializer for non-primitive types in JSON dumps."""
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return obj.model_dump(mode="python")
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return str(obj)


def _validate_page(limit: int, offset: int) -> None:
    """Reject pagination values that are not sane non-boolean integers."""
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_HISTORY_LIMIT
    ):
        raise ValueError(
            f"limit must be an integer between 1 and {MAX_HISTORY_LIMIT}, got {limit!r}"
        )
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or not 0 <= offset <= _SQLITE_MAX_INTEGER
    ):
        raise ValueError(
            f"offset must be an integer between 0 and {_SQLITE_MAX_INTEGER}, "
            f"got {offset!r}"
        )


_HISTORY_QUERY = """
SELECT id, method_name, timestamp
FROM flow_states
WHERE flow_uuid = ?
ORDER BY id ASC
LIMIT ? OFFSET ?
"""


def _fetch_history(
    conn: sqlite3.Connection, flow_uuid: str, limit: int, offset: int
) -> list[FlowStateHistoryRecord]:
    """Run the history query on an open connection and map rows to records.

    Timestamps written by older releases are naive UTC; they are normalised to
    aware UTC so records from one database always compare with each other.
    """
    rows = conn.execute(_HISTORY_QUERY, (flow_uuid, limit, offset)).fetchall()
    records: list[FlowStateHistoryRecord] = []
    for row_id, method_name, ts in rows:
        timestamp = datetime.fromisoformat(ts)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        records.append(
            FlowStateHistoryRecord(
                id=row_id, method_name=method_name, timestamp=timestamp
            )
        )
    return records


def _fetch_history_uri(
    uri: str, flow_uuid: str, limit: int, offset: int
) -> list[FlowStateHistoryRecord]:
    """Fetch history over a URI-opened (read-only or immutable) connection."""
    with closing(
        sqlite3.connect(uri, uri=True, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
    ) as conn:
        return _fetch_history(conn, flow_uuid, limit, offset)


def read_flow_state_history(
    db_path: str | Path,
    flow_uuid: str,
    *,
    limit: int = DEFAULT_HISTORY_LIMIT,
    offset: int = 0,
) -> list[FlowStateHistoryRecord]:
    """List state snapshot metadata for a flow without modifying the database.

    The database is opened read-only: no database file or table is created
    and no schema, journal mode, or stored data is changed. This makes it safe
    for inspection tools such as the CLI. State payloads are not loaded.

    Opening a WAL-mode database read-only may create ``-wal``/``-shm`` sidecar
    files next to it. If that fails (for example because the directory is not
    writable) and no ``-wal`` or ``-journal`` file exists, the main file holds
    every committed row and is read as immutable instead. If either sidecar
    exists (committed rows may still be in the WAL, or a hot journal still
    needs rolling back), or the failure is a lock contention, the error is
    raised rather than returning a history that may be incomplete or include
    uncommitted rows. An immutable read disables locking and change
    detection, so the database must not change from the sidecar check until
    the read completes; this implementation does not synchronise the two and
    offers no protection against a concurrent writer in that window.

    Args:
        db_path: Path to a ``SQLiteFlowPersistence`` database file.
        flow_uuid: Unique identifier for the flow instance.
        limit: Maximum number of records to return (1 to ``MAX_HISTORY_LIMIT``).
        offset: Number of records to skip.

    Returns:
        Records ordered oldest first by row ID, or an empty list if the
        database file, table, or flow does not exist.

    Raises:
        ValueError: If ``limit`` or ``offset`` is out of range.
        sqlite3.OperationalError: If the database cannot be opened read-only.
    """
    _validate_page(limit, offset)
    path = Path(db_path)
    if not path.exists():
        return []

    resolved = path.resolve()
    base_uri = resolved.as_uri()
    try:
        try:
            return _fetch_history_uri(f"{base_uri}?mode=ro", flow_uuid, limit, offset)
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if (
                "no such table" in message
                or "locked" in message
                or Path(f"{resolved}-wal").exists()
                or Path(f"{resolved}-journal").exists()
            ):
                raise
            return _fetch_history_uri(
                f"{base_uri}?immutable=1", flow_uuid, limit, offset
            )
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return []
        raise


class SQLiteFlowPersistence(FlowPersistence):
    """SQLite-based implementation of flow state persistence.

    This class provides a simple, file-based persistence implementation using SQLite.
    It's suitable for development and testing, or for production use cases with
    moderate performance requirements.

    This implementation supports async human feedback by storing pending feedback
    context in a separate table. When a flow is paused waiting for feedback,
    use save_pending_feedback() to persist the context. Later, use
    load_pending_feedback() to retrieve it when resuming.

    Example:
        ```python
        persistence = SQLiteFlowPersistence("flows.db")

        # Start a flow with async feedback
        try:
            flow = MyFlow(persistence=persistence)
            result = flow.kickoff()
        except HumanFeedbackPending as e:
            # Flow is paused, state is already persisted
            print(f"Waiting for feedback: {e.context.flow_id}")

        # Later, resume with feedback
        flow = MyFlow.from_pending("abc-123", persistence)
        result = flow.resume("looks good!")
        ```
    """

    persistence_type: str = Field(default="SQLiteFlowPersistence")
    db_path: str = Field(
        default_factory=lambda: str(Path(db_storage_path()) / "flow_states.db")
    )
    _lock_name: str = PrivateAttr()

    def __init__(self, db_path: str | None = None, /, **kwargs: Any) -> None:
        if db_path is not None:
            kwargs["db_path"] = db_path
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def _setup(self) -> Self:
        self._lock_name = f"sqlite:{os.path.realpath(self.db_path)}"
        self.init_db()
        return self

    def init_db(self) -> None:
        """Create the necessary tables if they don't exist."""
        deadline = monotonic() + _SQLITE_BUSY_TIMEOUT_SECONDS
        while True:
            try:
                self._create_schema()
                return
            except sqlite3.OperationalError as exc:  # noqa: PERF203
                if "locked" not in str(exc).lower() or monotonic() >= deadline:
                    raise
                sleep(_SCHEMA_RETRY_DELAY_SECONDS)

    def _create_schema(self) -> None:
        with (
            closing(
                sqlite3.connect(self.db_path, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
            ) as conn,
            conn,
        ):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS flow_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_uuid TEXT NOT NULL,
                method_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                state_json TEXT NOT NULL
            )
            """
            )
            conn.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_flow_states_uuid
            ON flow_states(flow_uuid)
            """
            )

            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS pending_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_uuid TEXT NOT NULL UNIQUE,
                context_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
            """
            )
            conn.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_pending_feedback_uuid
            ON pending_feedback(flow_uuid)
            """
            )

    def _save_state_sql(
        self,
        conn: sqlite3.Connection,
        flow_uuid: str,
        method_name: str,
        state_dict: dict[str, Any],
    ) -> None:
        """Execute the save-state INSERT without acquiring the lock.

        Args:
            conn: An open SQLite connection.
            flow_uuid: Unique identifier for the flow instance.
            method_name: Name of the method that just completed.
            state_dict: State data as a plain dict.
        """
        conn.execute(
            """
            INSERT INTO flow_states (
                flow_uuid,
                method_name,
                timestamp,
                state_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                flow_uuid,
                method_name,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(state_dict, default=_json_default),
            ),
        )

    @staticmethod
    def _to_state_dict(state_data: dict[str, Any] | BaseModel) -> dict[str, Any]:
        """Convert state_data to a plain dict."""
        if isinstance(state_data, BaseModel):
            try:
                return state_data.model_dump(mode="json")
            except Exception:
                return state_data.model_dump(mode="python")
        if isinstance(state_data, dict):
            return state_data
        raise ValueError(
            f"state_data must be either a Pydantic BaseModel or dict, got {type(state_data)}"
        )

    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        """Save the current flow state to SQLite.

        Args:
            flow_uuid: Unique identifier for the flow instance
            method_name: Name of the method that just completed
            state_data: Current state data (either dict or Pydantic model)
        """
        state_dict = self._to_state_dict(state_data)

        with (
            store_lock(self._lock_name),
            closing(
                sqlite3.connect(self.db_path, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
            ) as conn,
            conn,
        ):
            self._save_state_sql(conn, flow_uuid, method_name, state_dict)

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        """Load the most recent state for a given flow UUID.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            The most recent state as a dictionary, or None if no state exists
        """
        with (
            closing(
                sqlite3.connect(self.db_path, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
            ) as conn,
            conn,
        ):
            cursor = conn.execute(
                """
            SELECT state_json
            FROM flow_states
            WHERE flow_uuid = ?
            ORDER BY id DESC
            LIMIT 1
            """,
                (flow_uuid,),
            )
            row = cursor.fetchone()

        if row:
            result = json.loads(row[0])
            return result if isinstance(result, dict) else None
        return None

    def list_state_history(
        self,
        flow_uuid: str,
        *,
        limit: int = DEFAULT_HISTORY_LIMIT,
        offset: int = 0,
    ) -> list[FlowStateHistoryRecord]:
        """List metadata for every persisted state snapshot of a flow.

        Uses the same connection settings as :meth:`load_state`. To inspect a
        database without constructing a persistence instance (and so without
        initializing its schema), use :func:`read_flow_state_history`.

        Args:
            flow_uuid: Unique identifier for the flow instance.
            limit: Maximum number of records to return (1 to ``MAX_HISTORY_LIMIT``).
            offset: Number of records to skip.

        Returns:
            Records ordered oldest first by row ID.

        Raises:
            ValueError: If ``limit`` or ``offset`` is out of range.
        """
        _validate_page(limit, offset)
        with (
            closing(
                sqlite3.connect(self.db_path, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
            ) as conn,
            conn,
        ):
            return _fetch_history(conn, flow_uuid, limit, offset)

    def save_pending_feedback(
        self,
        flow_uuid: str,
        context: PendingFeedbackContext,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        """Save state with a pending feedback marker.

        This method stores both the flow state and the pending feedback context,
        allowing the flow to be resumed later when feedback is received.

        Args:
            flow_uuid: Unique identifier for the flow instance
            context: The pending feedback context with all resume information
            state_data: Current state data
        """
        state_dict = self._to_state_dict(state_data)

        with (
            store_lock(self._lock_name),
            closing(
                sqlite3.connect(self.db_path, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
            ) as conn,
            conn,
        ):
            self._save_state_sql(conn, flow_uuid, context.method_name, state_dict)

            conn.execute(
                """
            INSERT OR REPLACE INTO pending_feedback (
                flow_uuid,
                context_json,
                state_json,
                created_at
            ) VALUES (?, ?, ?, ?)
            """,
                (
                    flow_uuid,
                    json.dumps(context.to_dict(), default=_json_default),
                    json.dumps(state_dict, default=_json_default),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_pending_feedback(
        self,
        flow_uuid: str,
    ) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
        """Load state and pending feedback context.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            Tuple of (state_data, pending_context) if pending feedback exists,
            None otherwise.
        """
        # Import here to avoid circular imports
        from crewai.flow.async_feedback.types import PendingFeedbackContext

        with (
            closing(
                sqlite3.connect(self.db_path, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
            ) as conn,
            conn,
        ):
            cursor = conn.execute(
                """
            SELECT state_json, context_json
            FROM pending_feedback
            WHERE flow_uuid = ?
            """,
                (flow_uuid,),
            )
            row = cursor.fetchone()

        if row:
            state_dict = json.loads(row[0])
            context_dict = json.loads(row[1])
            context = PendingFeedbackContext.from_dict(context_dict)
            return (state_dict, context)
        return None

    def clear_pending_feedback(self, flow_uuid: str) -> None:
        """Clear the pending feedback marker after successful resume.

        Args:
            flow_uuid: Unique identifier for the flow instance
        """
        with (
            store_lock(self._lock_name),
            closing(
                sqlite3.connect(self.db_path, timeout=_SQLITE_BUSY_TIMEOUT_SECONDS)
            ) as conn,
            conn,
        ):
            conn.execute(
                """
            DELETE FROM pending_feedback
            WHERE flow_uuid = ?
            """,
                (flow_uuid,),
            )
