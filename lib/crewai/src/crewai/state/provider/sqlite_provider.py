"""SQLite state provider for checkpointing."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
import uuid

import aiosqlite

from crewai.state.provider.core import BaseProvider


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
)
"""

_INSERT = "INSERT INTO checkpoints (id, created_at, data) VALUES (?, ?, ?)"
_SELECT = "SELECT data FROM checkpoints WHERE id = ?"
_PRUNE = """
DELETE FROM checkpoints WHERE rowid NOT IN (
    SELECT rowid FROM checkpoints ORDER BY rowid DESC LIMIT ?
)
"""


def _make_id() -> tuple[str, str]:
    """Generate a checkpoint ID and ISO timestamp.

    Returns:
        A tuple of (checkpoint_id, timestamp).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    checkpoint_id = f"{ts}_{uuid.uuid4().hex[:8]}"
    return checkpoint_id, ts


class SqliteProvider(BaseProvider):
    """Persists runtime state checkpoints in a SQLite database.

    The ``directory`` argument to ``checkpoint`` / ``acheckpoint`` is
    used as the database path (e.g. ``"./.checkpoints.db"``).

    Args:
        max_checkpoints: Maximum number of checkpoints to retain.
            Oldest rows are pruned after each write. None keeps all.
    """

    def __init__(self, max_checkpoints: int | None = None) -> None:
        self.max_checkpoints = max_checkpoints

    def checkpoint(self, data: str, directory: str) -> str:
        """Write a checkpoint to the SQLite database.

        Args:
            data: The serialized JSON string to persist.
            directory: Path to the SQLite database file.

        Returns:
            A location string in the format ``"db_path#checkpoint_id"``.
        """
        checkpoint_id, ts = _make_id()
        conn = sqlite3.connect(directory)
        try:
            conn.execute(_CREATE_TABLE)
            conn.execute(_INSERT, (checkpoint_id, ts, data))
            if self.max_checkpoints is not None:
                conn.execute(_PRUNE, (self.max_checkpoints,))
            conn.commit()
        finally:
            conn.close()
        return f"{directory}#{checkpoint_id}"

    async def acheckpoint(self, data: str, directory: str) -> str:
        """Write a checkpoint to the SQLite database asynchronously.

        Args:
            data: The serialized JSON string to persist.
            directory: Path to the SQLite database file.

        Returns:
            A location string in the format ``"db_path#checkpoint_id"``.
        """
        checkpoint_id, ts = _make_id()
        async with aiosqlite.connect(directory) as db:
            await db.execute(_CREATE_TABLE)
            await db.execute(_INSERT, (checkpoint_id, ts, data))
            if self.max_checkpoints is not None:
                await db.execute(_PRUNE, (self.max_checkpoints,))
            await db.commit()
        return f"{directory}#{checkpoint_id}"

    def from_checkpoint(self, location: str) -> str:
        """Read a checkpoint from the SQLite database.

        Args:
            location: A location string returned by ``checkpoint()``.

        Returns:
            The raw JSON string.

        Raises:
            ValueError: If the checkpoint ID is not found.
        """
        db_path, checkpoint_id = location.rsplit("#", 1)
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(_SELECT, (checkpoint_id,)).fetchone()
            if row is None:
                raise ValueError(f"Checkpoint not found: {checkpoint_id}")
            result: str = row[0]
            return result
        finally:
            conn.close()

    async def afrom_checkpoint(self, location: str) -> str:
        """Read a checkpoint from the SQLite database asynchronously.

        Args:
            location: A location string returned by ``acheckpoint()``.

        Returns:
            The raw JSON string.

        Raises:
            ValueError: If the checkpoint ID is not found.
        """
        db_path, checkpoint_id = location.rsplit("#", 1)
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(_SELECT, (checkpoint_id,))
            row = await cursor.fetchone()
            if row is None:
                raise ValueError(f"Checkpoint not found: {checkpoint_id}")
            result: str = row[0]
            return result
