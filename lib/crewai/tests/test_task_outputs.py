"""Connection-lifecycle tests for the ``crewai log-tasks-outputs`` reader."""

from __future__ import annotations

from contextlib import closing
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from crewai_cli import task_outputs as task_outputs_module
from crewai_cli.task_outputs import load_task_outputs


def _create_task_outputs_db(db_path: Path) -> None:
    """Build the table ``KickoffTaskOutputsSQLiteStorage`` writes."""
    # Close the setup handle too: a leaked one holds the database lock on Windows, which would
    # make the file-release assertions below fail for the fixture's reason, not the reader's.
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            """CREATE TABLE latest_kickoff_task_outputs (
                task_id TEXT PRIMARY KEY,
                task_key TEXT,
                expected_output TEXT,
                output JSON,
                task_index INTEGER,
                inputs JSON,
                was_replayed BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.execute(
            "INSERT INTO latest_kickoff_task_outputs "
            "(task_id, task_key, expected_output, output, task_index, inputs, "
            "was_replayed, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "task-1",
                "task-1-key",
                "A summary",
                '{"raw": "done"}',
                0,
                '{"topic": "ai"}',
                0,
                "2026-01-01T00:00:00",
            ),
        )
        conn.commit()


def test_load_task_outputs_reads_rows(tmp_path: Path) -> None:
    """Sanity check: the reader still returns the stored rows."""
    db_path = tmp_path / "outputs.db"
    _create_task_outputs_db(db_path)

    rows = load_task_outputs(str(db_path))

    assert len(rows) == 1
    assert rows[0]["task_id"] == "task-1"
    assert rows[0]["output"] == {"raw": "done"}
    assert rows[0]["inputs"] == {"topic": "ai"}


def test_load_task_outputs_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader closes its connection instead of leaking the handle.

    ``with sqlite3.connect(...) as conn`` only commits or rolls back; it never
    closes, so the handle outlives the call and keeps the database locked on
    Windows until a cyclic GC pass.
    """
    db_path = tmp_path / "outputs.db"
    _create_task_outputs_db(db_path)

    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(task_outputs_module.sqlite3, "connect", tracking_connect)

    assert load_task_outputs(str(db_path))

    assert opened, "expected the reader to open a connection"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 1")


def test_database_file_is_releasable_after_read(tmp_path: Path) -> None:
    """The database can be removed straight away, without a GC pass."""
    db_path = tmp_path / "outputs.db"
    _create_task_outputs_db(db_path)

    load_task_outputs(str(db_path))

    db_path.unlink()
    assert not db_path.exists()
