"""Storage-format regression tests for the SQLite checkpoint provider.

The provider used SQLite's ``jsonb()`` when writing, which exists only in
SQLite >= 3.45.0 and stored a binary blob. On system Pythons shipping older
SQLite (Ubuntu 22.04, Debian 12, RHEL 9, macOS) every checkpoint write failed
with ``no such function: jsonb``. Rows are now stored as plain JSON text and
read back without any JSON1 function; rows written as JSONB blobs by the
previous writer stay readable on the SQLite versions that can decode them.
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


def test_from_checkpoint_reads_rows_written_without_json_functions(
    tmp_path: Path,
) -> None:
    """Text rows are readable without any JSON1 function, as on old SQLite."""
    provider = SqliteProvider()
    db_path = tmp_path / "checkpoints.db"
    provider.checkpoint('{"step": 1}', str(db_path))

    # A plain parameterized INSERT needs no JSON functions, so this is
    # exactly what any SQLite version can write and read back.
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO checkpoints (id, created_at, parent_id, branch, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "20260101T000000_plain",
                "20260101T000000",
                None,
                "main",
                '{"step": 0}',
            ),
        )

    restored = provider.from_checkpoint(f"{db_path}#20260101T000000_plain")
    assert json.loads(restored) == {"step": 0}


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < (3, 45, 0),
    reason="legacy JSONB blobs can only be written and decoded by SQLite >= 3.45",
)
def test_from_checkpoint_reads_legacy_jsonb_rows(tmp_path: Path) -> None:
    """Checkpoint databases written by the jsonb() writer remain readable.

    Reading legacy JSONB blobs also requires SQLite >= 3.45 — the only
    versions that can write them — which the skipif above already asserts.
    """
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


def test_legacy_jsonb_row_under_old_sqlite_raises_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy blob row on SQLite < 3.45 fails with an actionable error."""
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 40, 1))

    provider = SqliteProvider()
    db_path = tmp_path / "checkpoints.db"
    provider.checkpoint('{"step": 1}', str(db_path))

    # Legacy rows are binary JSONB blobs. The version check fires before any
    # decoding, so a BLOB bound directly (no jsonb() call) exercises the same
    # path — and keeps the test runnable on SQLite engines < 3.45 too. Real
    # JSONB blobs are covered by test_from_checkpoint_reads_legacy_jsonb_rows.
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO checkpoints (id, created_at, parent_id, branch, data) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "20260101T000000_legacy",
                "20260101T000000",
                None,
                "main",
                b"legacy-jsonb-blob",
            ),
        )

    with pytest.raises(ValueError, match="SQLite >= 3.45.0"):
        provider.from_checkpoint(f"{db_path}#20260101T000000_legacy")


def test_checkpoint_round_trip_after_fix(tmp_path: Path) -> None:
    """A checkpoint written as text restores identically."""
    provider = SqliteProvider()
    db_path = tmp_path / "checkpoints.db"

    payload = json.dumps({"flow": "x", "steps": [1, 2, 3], "nested": {"a": True}})
    location = provider.checkpoint(payload, str(db_path))
    restored = provider.from_checkpoint(location)

    assert json.loads(restored) == json.loads(payload)


@pytest.mark.asyncio
async def test_acheckpoint_round_trip_after_fix(tmp_path: Path) -> None:
    """The async path also writes text rows and restores them."""
    provider = SqliteProvider()
    db_path = tmp_path / "checkpoints.db"

    payload = json.dumps({"flow": "async", "ok": True})
    location = await provider.acheckpoint(payload, str(db_path))
    restored = await provider.afrom_checkpoint(location)

    assert json.loads(restored) == json.loads(payload)

    with closing(sqlite3.connect(db_path)) as conn:
        stored_type = conn.execute(
            "SELECT typeof(data) FROM checkpoints WHERE id = ?",
            (provider.extract_id(location),),
        ).fetchone()
    assert stored_type is not None and stored_type[0] == "text"
