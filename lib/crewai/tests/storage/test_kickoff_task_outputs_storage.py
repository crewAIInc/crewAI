"""Tests for ``KickoffTaskOutputsSQLiteStorage`` connection lifecycle."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from crewai.memory.storage import kickoff_task_outputs_storage as storage_module
from crewai.memory.storage.kickoff_task_outputs_storage import (
    KickoffTaskOutputsSQLiteStorage,
)
from crewai.task import Task


def _make_task() -> Task:
    """Build a minimal task whose id and key are enough for the storage layer."""
    return Task(description="Summarise the report", expected_output="A summary")


def _exercise(storage: KickoffTaskOutputsSQLiteStorage, task: Task) -> None:
    """Run every storage operation once."""
    storage.add(task, {"raw": "done"}, task_index=0, inputs={"topic": "ai"})
    storage.update(0, output={"raw": "updated"})
    assert storage.load()[0]["output"] == {"raw": "updated"}
    storage.delete_all()


def test_every_connection_is_closed_after_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each operation closes the connection it opened instead of leaking it.

    ``with sqlite3.connect(...) as conn`` only commits or rolls back; it never
    closes. The connection then survives in a reference cycle until the cyclic
    garbage collector runs, keeping the database file open (and, on Windows,
    locked) long after the call returned.
    """
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    monkeypatch.setattr(storage_module.sqlite3, "connect", tracking_connect)

    storage = KickoffTaskOutputsSQLiteStorage(db_path=str(tmp_path / "outputs.db"))
    _exercise(storage, _make_task())

    assert len(opened) == 5  # init, add, update, load, delete_all
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 1")


def test_database_file_is_not_locked_after_use(tmp_path: Path) -> None:
    """The database file can be removed right after use, without a GC pass.

    This is the user-visible symptom on Windows, where an open handle blocks
    ``os.remove``/``os.replace`` with ``PermissionError`` (WinError 32).
    """
    db_path = tmp_path / "outputs.db"
    storage = KickoffTaskOutputsSQLiteStorage(db_path=str(db_path))
    _exercise(storage, _make_task())
    del storage

    os.remove(db_path)
    assert not db_path.exists()
