"""SQLite-based cache backend for cross-process idempotent deduplication."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from crewai_core.lock_store import lock as store_lock
from crewai_core.paths import db_storage_path


_DEFAULT_DB_NAME = "cache_handler.db"


class SQLiteCacheBackend:
    """Persistent cache backend backed by SQLite.

    Uses file-level locking (via crewai_core.lock_store) so that multiple
    processes writing to the same database file correctly serialize access.

    Args:
        db_path: Full path to the SQLite database file. Defaults to
            ``<db_storage_path>/cache_handler.db``.
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = str(Path(db_storage_path()) / _DEFAULT_DB_NAME)
        self._db_path = db_path
        self._lock_name = f"cache_backend_{Path(db_path).stem}"
        self._init_db()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with store_lock(self._lock_name):
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tool_cache (
                        key   TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Protocol methods
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        with store_lock(self._lock_name):
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM tool_cache WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    return None
                return json.loads(row[0])
            finally:
                conn.close()

    def set(self, key: str, value: Any) -> None:
        with store_lock(self._lock_name):
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO tool_cache (key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (key, json.dumps(value)),
                )
                conn.commit()
            finally:
                conn.close()

    def claim_if_absent(self, key: str, sentinel: Any) -> tuple[bool, Any | None]:
        with store_lock(self._lock_name):
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value FROM tool_cache WHERE key = ?", (key,)
                ).fetchone()
                if row is not None:
                    return False, json.loads(row[0])
                conn.execute(
                    "INSERT INTO tool_cache (key, value) VALUES (?, ?)",
                    (key, json.dumps(sentinel)),
                )
                conn.commit()
                return True, None
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)
