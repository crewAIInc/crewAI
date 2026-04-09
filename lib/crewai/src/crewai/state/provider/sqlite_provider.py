"""SQLite state provider for checkpointing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Literal
import uuid

import aiosqlite

from crewai.state.provider.core import BaseProvider


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    parent_id TEXT,
    branch TEXT NOT NULL DEFAULT 'main',
    data JSONB NOT NULL
)
"""

_INSERT = (
    "INSERT INTO checkpoints (id, created_at, parent_id, branch, data) "
    "VALUES (?, ?, ?, ?, jsonb(?))"
)
_SELECT = "SELECT json(data) FROM checkpoints WHERE id = ?"
_PRUNE = """
DELETE FROM checkpoints WHERE branch = ? AND rowid NOT IN (
    SELECT rowid FROM checkpoints WHERE branch = ? ORDER BY rowid DESC LIMIT ?
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

    The ``location`` argument to ``checkpoint`` / ``acheckpoint`` is
    used as the database file path.
    """

    provider_type: Literal["sqlite"] = "sqlite"

    def checkpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """Write a checkpoint to the SQLite database.

        Args:
            data: The serialized JSON string to persist.
            location: Path to the SQLite database file.
            parent_id: ID of the parent checkpoint for lineage tracking.
            branch: Branch label for this checkpoint.

        Returns:
            A location string in the format ``"db_path#checkpoint_id"``.
        """
        checkpoint_id, ts = _make_id()
        Path(location).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(location) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE)
            conn.execute(_INSERT, (checkpoint_id, ts, parent_id, branch, data))
            conn.commit()
        return f"{location}#{checkpoint_id}"

    async def acheckpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """Write a checkpoint to the SQLite database asynchronously.

        Args:
            data: The serialized JSON string to persist.
            location: Path to the SQLite database file.
            parent_id: ID of the parent checkpoint for lineage tracking.
            branch: Branch label for this checkpoint.

        Returns:
            A location string in the format ``"db_path#checkpoint_id"``.
        """
        checkpoint_id, ts = _make_id()
        Path(location).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(location) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(_CREATE_TABLE)
            await db.execute(_INSERT, (checkpoint_id, ts, parent_id, branch, data))
            await db.commit()
        return f"{location}#{checkpoint_id}"

    def prune(self, location: str, max_keep: int, *, branch: str = "main") -> None:
        """Remove oldest checkpoint rows beyond *max_keep* on a branch."""
        with sqlite3.connect(location) as conn:
            conn.execute(_PRUNE, (branch, branch, max_keep))
            conn.commit()

    def extract_id(self, location: str) -> str:
        """Extract the checkpoint ID from a ``db_path#id`` string."""
        return location.rsplit("#", 1)[1]

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
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(_SELECT, (checkpoint_id,)).fetchone()
            if row is None:
                raise ValueError(f"Checkpoint not found: {checkpoint_id}")
            result: str = row[0]
            return result

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
