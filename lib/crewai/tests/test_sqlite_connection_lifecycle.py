"""Connection-lifecycle tests for the flow persistence and checkpoint SQLite backends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier

import pytest
from crewai.flow.async_feedback.types import PendingFeedbackContext
from crewai.flow.persistence import sqlite as sqlite_module
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from crewai.state.provider.sqlite_provider import SqliteProvider


_CONCURRENT_START_TIMEOUT_SECONDS = 10
_CHILD_PROCESS_TIMEOUT_SECONDS = 60


def _track_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> list[sqlite3.Connection]:
    """Record every connection ``sqlite3.connect`` hands out for the rest of the test."""
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracking_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    return opened


def _assert_all_closed(opened: list[sqlite3.Connection]) -> None:
    """Every recorded connection must reject further use."""
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 1")


def _make_context() -> PendingFeedbackContext:
    """Build a minimal pending-feedback context."""
    return PendingFeedbackContext(
        flow_id="flow-1",
        flow_class="tests.ReviewFlow",
        method_name="review",
        method_output="draft",
        message="Please review",
    )


class _FailingConnection(sqlite3.Connection):
    """Connection whose ``execute`` fails on INSERT, after any preceding statements."""

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:  # type: ignore[override]
        if sql.lstrip().upper().startswith("INSERT"):
            raise sqlite3.OperationalError("simulated failure inside the transaction")
        return super().execute(sql, *args)  # type: ignore[arg-type]


def test_flow_persistence_closes_every_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each ``SQLiteFlowPersistence`` operation closes the connection it opened."""
    opened = _track_connections(monkeypatch)

    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))  # init_db
    persistence.save_state("flow-1", "start", {"step": 1})
    assert persistence.load_state("flow-1") == {"step": 1}
    persistence.save_pending_feedback("flow-1", _make_context(), {"step": 2})
    loaded = persistence.load_pending_feedback("flow-1")
    assert loaded is not None and loaded[0] == {"step": 2}
    persistence.clear_pending_feedback("flow-1")

    assert len(opened) == 6
    _assert_all_closed(opened)


def test_flow_persistence_db_is_removable_after_use(tmp_path: Path) -> None:
    """The flow database can be deleted right after use, without a GC pass."""
    db_path = tmp_path / "flows.db"
    persistence = SQLiteFlowPersistence(str(db_path))
    persistence.save_state("flow-1", "start", {"step": 1})
    persistence.load_state("flow-1")
    del persistence

    os.remove(db_path)
    assert not db_path.exists()


def test_flow_persistence_failed_write_rolls_back_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing INSERT inside ``save_state`` rolls back and still closes the connection."""
    persistence = SQLiteFlowPersistence(str(tmp_path / "flows.db"))

    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def failing_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        kwargs["factory"] = _FailingConnection
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", failing_connect)
    with pytest.raises(sqlite3.OperationalError):
        persistence.save_state("flow-1", "start", {"step": 1})
    monkeypatch.undo()

    assert len(opened) == 1
    _assert_all_closed(opened)
    assert persistence.load_state("flow-1") is None


@pytest.mark.parametrize("existing_db", [False, True])
def test_parallel_flow_initialization_does_not_use_store_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_db: bool
) -> None:
    db_path = tmp_path / "flows.db"
    if existing_db:
        persistence = SQLiteFlowPersistence(str(db_path))
        persistence.save_state("prior-flow", "start", {"step": 1})
        persistence.save_pending_feedback("prior-flow", _make_context(), {"step": 2})

    def reject_store_lock(*args: object, **kwargs: object) -> None:
        raise AssertionError("initialization must not acquire a shared store lock")

    monkeypatch.setattr(sqlite_module, "store_lock", reject_store_lock)
    barrier = Barrier(16)

    def initialize(_: int) -> SQLiteFlowPersistence:
        barrier.wait(timeout=_CONCURRENT_START_TIMEOUT_SECONDS)
        return SQLiteFlowPersistence(str(db_path))

    with ThreadPoolExecutor(max_workers=16) as pool:
        persistence_instances = list(pool.map(initialize, range(16)))

    assert len(persistence_instances) == 16
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        } >= {"flow_states", "pending_feedback"}
    if existing_db:
        assert persistence_instances[0].load_state("prior-flow") == {"step": 2}
        pending = persistence_instances[0].load_pending_feedback("prior-flow")
        assert pending is not None and pending[0] == {"step": 2}


@pytest.mark.parametrize("existing_db", [False, True])
def test_parallel_processes_initialize_flow_db(
    tmp_path: Path, existing_db: bool
) -> None:
    db_path = tmp_path / "flows.db"
    if existing_db:
        persistence = SQLiteFlowPersistence(str(db_path))
        persistence.save_state("prior-flow", "start", {"step": 1})

    barrier = Barrier(2)
    script = (
        "import sys; from crewai.flow.persistence.sqlite import SQLiteFlowPersistence; "
        "SQLiteFlowPersistence(sys.argv[1])"
    )

    def initialize(_: int) -> None:
        barrier.wait(timeout=_CONCURRENT_START_TIMEOUT_SECONDS)
        subprocess.run(
            [sys.executable, "-c", script, str(db_path)],
            check=True,
            timeout=_CHILD_PROCESS_TIMEOUT_SECONDS,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(initialize, range(2)))

    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    persistence = SQLiteFlowPersistence(str(db_path))
    if existing_db:
        assert persistence.load_state("prior-flow") == {"step": 1}
    persistence.save_state("new-flow", "start", {"step": 2})
    assert persistence.load_state("new-flow") == {"step": 2}


def test_sqlite_provider_closes_every_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each ``SqliteProvider`` operation closes the connection it opened."""
    opened = _track_connections(monkeypatch)
    db_path = str(tmp_path / "checkpoints.db")

    provider = SqliteProvider()
    location = provider.checkpoint('{"a": 1}', db_path, branch="main")
    # Stored as jsonb, so the text comes back normalised; compare the parsed value.
    assert json.loads(provider.from_checkpoint(location)) == {"a": 1}
    provider.prune(db_path, 1, branch="main")

    assert len(opened) == 3
    _assert_all_closed(opened)


def test_sqlite_provider_db_is_removable_after_use(tmp_path: Path) -> None:
    """The checkpoint database can be deleted right after use, without a GC pass."""
    db_path = tmp_path / "checkpoints.db"
    provider = SqliteProvider()
    location = provider.checkpoint('{"a": 1}', str(db_path), branch="main")
    provider.from_checkpoint(location)

    os.remove(db_path)
    assert not db_path.exists()
