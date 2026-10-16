"""Tests for ``KickoffTaskOutputsSQLiteStorage``."""

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
from crewai.utilities.errors import DatabaseOperationError


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


class _FailingCursor(sqlite3.Cursor):
    """Cursor that fails on INSERT, after the storage has already issued BEGIN."""

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:  # type: ignore[override]
        if sql.lstrip().upper().startswith("INSERT"):
            raise sqlite3.OperationalError("simulated failure after BEGIN")
        return super().execute(sql, *args)  # type: ignore[arg-type]


class _FailingConnection(sqlite3.Connection):
    """Connection whose ``cursor()`` hands out ``_FailingCursor`` instances."""

    def cursor(self, factory: type[sqlite3.Cursor] = _FailingCursor) -> sqlite3.Cursor:  # type: ignore[override]
        return super().cursor(factory)


def test_failed_write_rolls_back_and_closes_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure between BEGIN and COMMIT rolls back and still closes the connection.

    ``with closing(...) as conn, conn:`` must roll back on the inner context
    manager and close on the outer one even when the operation raises.
    """
    storage = KickoffTaskOutputsSQLiteStorage(db_path=str(tmp_path / "outputs.db"))

    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def failing_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = _FailingConnection
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    monkeypatch.setattr(storage_module.sqlite3, "connect", failing_connect)
    with pytest.raises(DatabaseOperationError):
        storage.add(_make_task(), {"raw": "done"}, task_index=0)
    monkeypatch.undo()

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")
    assert storage.load() == []


@pytest.mark.parametrize(
    "output",
    [["result_1", "result_2"], "plain text", 42, {"raw": "updated"}],
)
def test_update_round_trips_json_values(tmp_path: Path, output: object) -> None:
    """``update`` stores any JSON value in ``output``, not just dicts."""
    storage = KickoffTaskOutputsSQLiteStorage(db_path=str(tmp_path / "outputs.db"))
    storage.add(_make_task(), {"raw": "done"}, task_index=0)

    storage.update(0, output=output, inputs=["a", "b"])

    row = storage.load()[0]
    assert row["output"] == output
    assert row["inputs"] == ["a", "b"]


def test_load_returns_none_for_null_json_columns(tmp_path: Path) -> None:
    """A NULL ``output`` or ``inputs`` loads as ``None`` instead of crashing."""
    storage = KickoffTaskOutputsSQLiteStorage(db_path=str(tmp_path / "outputs.db"))
    storage.add(_make_task(), {"raw": "done"}, task_index=0)

    storage.update(0, output=None, inputs=None)

    row = storage.load()[0]
    assert row["output"] is None
    assert row["inputs"] is None


def test_load_wraps_malformed_json_in_database_error(tmp_path: Path) -> None:
    """Undecodable stored JSON raises ``DatabaseOperationError``."""
    db_path = tmp_path / "outputs.db"
    storage = KickoffTaskOutputsSQLiteStorage(db_path=str(db_path))
    storage.add(_make_task(), {"raw": "done"}, task_index=0)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE latest_kickoff_task_outputs SET output = 'not json'")
    conn.close()

    with pytest.raises(DatabaseOperationError):
        storage.load()
