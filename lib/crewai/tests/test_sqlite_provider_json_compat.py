"""Storage-format regression tests for the SQLite checkpoint provider.

The provider used SQLite's ``jsonb()`` when writing, which exists only in
SQLite >= 3.45.0 and stored a binary blob. On system Pythons shipping older
SQLite (Ubuntu 22.04, Debian 12, RHEL 9, macOS) every checkpoint write failed
with ``no such function: jsonb``. Rows must be stored as plain JSON text,
while rows written as JSONB blobs by the previous writer stay readable.
"""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest
from crewai.state.provider.sqlite_provider import SqliteProvider


def test_checkpoint_data_is_stored_as_text(tmp_path: Path) -> None:
    """New checkpoint rows are plain JSON text, not JSONB blobs."""
    provider = SqliteProvider()
    db_path = tmp_path / "checkpoints.db"
    location = provider.checkpoint('{"step": 1}', str(db_path))

    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT typeof(data), data FROM checkpoints WHERE id = ?",
            (provider.extract_id(location),),
        ).fetchone()

    assert row is not None
    stored_type, stored_value = row
    assert stored_type == "text"
    assert json.loads(stored_value) == {"step": 1}


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 45, 0),
    reason="legacy JSONB blobs can only be written by SQLite >= 3.45",
)
def test_from_checkpoint_reads_legacy_jsonb_rows(tmp_path: Path) -> None:
    """Checkpoint databases written by the jsonb() writer remain readable."""
    provider = SqliteProvider()
    db_path = tmp_path / "checkpoints.db"
    provider.checkpoint('{"step": 1}', str(db_path))

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO checkpoints (id, created_at, parent_id, branch, data) "
            "VALUES (?, ?, ?, ?, jsonb(?))",
            ("20260101T000000_legacy", "20260101T000000", None, "main", '{"step": 0}'),
        )

    restored = provider.from_checkpoint(f"{db_path}#20260101T000000_legacy")
    assert json.loads(restored) == {"step": 0}


def test_checkpoint_round_trip_after_fix(tmp_path: Path) -> None:
    """A checkpoint written as text restores identically."""
    provider = SqliteProvider()
    db_path = tmp_path / "checkpoints.db"

    payload = json.dumps({"flow": "x", "steps": [1, 2, 3], "nested": {"a": True}})
    location = provider.checkpoint(payload, str(db_path))
    restored = provider.from_checkpoint(location)

    assert json.loads(restored) == json.loads(payload)
