"""Tests for read-only, paginated flow state history in SQLite persistence."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import sys
from typing import Any

import pytest
from crewai.flow.async_feedback.types import PendingFeedbackContext
from crewai.flow.persistence import (
    MAX_HISTORY_LIMIT,
    FlowPersistence,
    FlowStateHistoryRecord,
    SQLiteFlowPersistence,
    read_flow_state_history,
)
from crewai.flow.persistence import sqlite as sqlite_module


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "flows.db")


@pytest.fixture
def persistence(db_path: str) -> SQLiteFlowPersistence:
    return SQLiteFlowPersistence(db_path)


def _methods(records: list[FlowStateHistoryRecord]) -> list[str]:
    return [r.method_name for r in records]


def test_history_is_scoped_per_flow_uuid_and_ordered_by_save(
    persistence: SQLiteFlowPersistence,
) -> None:
    persistence.save_state("flow-a", "a1", {"n": 1})
    persistence.save_state("flow-b", "b1", {"n": 1})
    persistence.save_state("flow-a", "a2", {"n": 2})
    persistence.save_state("flow-b", "b2", {"n": 2})
    persistence.save_state("flow-a", "a3", {"n": 3})

    history_a = persistence.list_state_history("flow-a")
    history_b = persistence.list_state_history("flow-b")

    assert _methods(history_a) == ["a1", "a2", "a3"]
    assert _methods(history_b) == ["b1", "b2"]
    assert [r.id for r in history_a] == sorted(r.id for r in history_a)
    assert all(isinstance(r.timestamp, datetime) for r in history_a)
    assert not {r.id for r in history_a} & {r.id for r in history_b}


def test_history_ids_are_stable_across_calls(
    persistence: SQLiteFlowPersistence,
) -> None:
    persistence.save_state("flow", "first", {})
    first_id = persistence.list_state_history("flow")[0].id

    persistence.save_state("flow", "second", {})

    assert persistence.list_state_history("flow")[0].id == first_id


def test_empty_history_for_unknown_flow(persistence: SQLiteFlowPersistence) -> None:
    persistence.save_state("other", "step", {})

    assert persistence.list_state_history("missing") == []


def test_empty_history_when_database_does_not_exist(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"

    assert read_flow_state_history(missing, "flow") == []
    assert not missing.exists()


def test_directory_path_is_not_treated_as_empty_history(tmp_path: Path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        read_flow_state_history(tmp_path, "flow")


def test_empty_history_when_table_does_not_exist(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
        conn.commit()

    assert read_flow_state_history(path, "flow") == []
    with closing(sqlite3.connect(path)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables == {"unrelated"}


def test_pagination_pages_through_full_history(
    persistence: SQLiteFlowPersistence,
) -> None:
    for i in range(7):
        persistence.save_state("flow", f"step{i}", {"i": i})

    pages = [
        persistence.list_state_history("flow", limit=3, offset=offset)
        for offset in (0, 3, 6)
    ]

    assert [len(page) for page in pages] == [3, 3, 1]
    assert _methods([r for page in pages for r in page]) == [
        f"step{i}" for i in range(7)
    ]
    assert persistence.list_state_history("flow", limit=3, offset=7) == []


def test_equal_timestamps_are_ordered_by_row_id(
    persistence: SQLiteFlowPersistence, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = datetime(2026, 1, 1, 12, 0, 0)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return frozen.replace(tzinfo=tz)

    with monkeypatch.context() as m:
        m.setattr(sqlite_module, "datetime", FrozenDatetime)
        for name in ("x", "y", "z"):
            persistence.save_state("flow", name, {})

    history = persistence.list_state_history("flow")

    assert len({r.timestamp for r in history}) == 1
    assert _methods(history) == ["x", "y", "z"]
    assert _methods(persistence.list_state_history("flow", limit=1, offset=1)) == [
        "y"
    ]


def test_naive_timestamps_are_normalised_to_utc(
    persistence: SQLiteFlowPersistence, db_path: str
) -> None:
    persistence.save_state("flow", "aware", {})
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "INSERT INTO flow_states (flow_uuid, method_name, timestamp, state_json)"
            " VALUES ('flow', 'naive', '2025-02-01T10:00:00', '{}')"
        )
        conn.commit()

    history = persistence.list_state_history("flow")

    assert _methods(history) == ["aware", "naive"]
    assert all(r.timestamp.tzinfo is not None for r in history)
    assert history[1].timestamp == datetime(2025, 2, 1, 10, 0, tzinfo=timezone.utc)
    # Mixed naive/aware rows would raise TypeError here.
    sorted(history, key=lambda r: r.timestamp)


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX file permissions are not enforced",
)
def test_reader_raises_instead_of_truncating_when_wal_is_unreadable(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src" / "flows.db"
    src.parent.mkdir()
    persistence = SQLiteFlowPersistence(str(src))
    persistence.save_state("flow", "step", {})
    # An open read transaction keeps committed rows in the WAL instead of
    # letting the writer checkpoint and delete it on close.
    holder = sqlite3.connect(src)
    holder.execute("BEGIN")
    holder.execute("SELECT COUNT(*) FROM flow_states").fetchone()
    persistence.save_state("flow", "in-wal", {})
    # Copy db + WAL (but no -shm) somewhere the reader cannot create a -shm.
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    shutil.copy(src, ro_dir / "flows.db")
    shutil.copy(src.with_suffix(".db-wal"), ro_dir / "flows.db-wal")
    holder.close()
    link = tmp_path / "link.db"
    link.symlink_to(ro_dir / "flows.db")
    dir_mode = ro_dir.stat().st_mode
    ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(sqlite3.OperationalError):
            read_flow_state_history(ro_dir / "flows.db", "flow")
        # The -wal check must follow symlinks to the real database.
        with pytest.raises(sqlite3.OperationalError):
            read_flow_state_history(link, "flow")
    finally:
        ro_dir.chmod(dir_mode)


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX file permissions are not enforced",
)
def test_reader_raises_instead_of_returning_uncommitted_rows_from_hot_journal(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src" / "flows.db"
    src.parent.mkdir()
    conn = sqlite3.connect(src, isolation_level=None)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA cache_size=1")  # spill dirty pages to the main file
    conn.execute(
        "CREATE TABLE flow_states (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " flow_uuid TEXT NOT NULL, method_name TEXT NOT NULL,"
        " timestamp DATETIME NOT NULL, state_json TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO flow_states (flow_uuid, method_name, timestamp, state_json)"
        " VALUES ('flow', 'committed', '2026-01-01T00:00:00+00:00', '{}')"
    )
    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO flow_states (flow_uuid, method_name, timestamp, state_json)"
        " VALUES ('flow', 'UNCOMMITTED', '2026-01-01T00:00:00+00:00', ?)",
        [("x" * 2000,)] * 2000,
    )
    # Copy db + hot rollback journal mid-transaction to simulate a crash.
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    shutil.copy(src, ro_dir / "flows.db")
    shutil.copy(src.with_suffix(".db-journal"), ro_dir / "flows.db-journal")
    conn.rollback()
    conn.close()
    dir_mode = ro_dir.stat().st_mode
    ro_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(sqlite3.OperationalError):
            read_flow_state_history(ro_dir / "flows.db", "flow")
    finally:
        ro_dir.chmod(dir_mode)


def test_history_does_not_read_state_payloads(
    persistence: SQLiteFlowPersistence,
    db_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence.save_state("flow", "step", {"big": "x" * 10_000})
    statements: list[str] = []
    real_connect = sqlite3.connect

    def tracing_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracing_connect)

    read_flow_state_history(db_path, "flow")

    assert statements
    assert not any("state_json" in sql for sql in statements)


@pytest.mark.parametrize(
    ("limit", "offset"),
    [
        (0, 0),
        (-1, 0),
        (MAX_HISTORY_LIMIT + 1, 0),
        (True, 0),
        (1.5, 0),
        (10, -1),
        (10, False),
        (10, "0"),
        (10, 2**63),
    ],
)
def test_invalid_pagination_is_rejected(
    persistence: SQLiteFlowPersistence, limit: Any, offset: Any
) -> None:
    with pytest.raises(ValueError):
        persistence.list_state_history("flow", limit=limit, offset=offset)


def test_max_limit_is_accepted(persistence: SQLiteFlowPersistence) -> None:
    persistence.save_state("flow", "step", {})

    history = persistence.list_state_history(
        "flow", limit=MAX_HISTORY_LIMIT
    )

    assert len(history) == 1


def test_max_offset_is_accepted(persistence: SQLiteFlowPersistence) -> None:
    persistence.save_state("flow", "step", {})

    assert persistence.list_state_history("flow", offset=2**63 - 1) == []


def test_reader_does_not_initialize_or_modify_database(tmp_path: Path) -> None:
    path = tmp_path / "plain.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE flow_states (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " flow_uuid TEXT NOT NULL, method_name TEXT NOT NULL,"
            " timestamp DATETIME NOT NULL, state_json TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO flow_states (flow_uuid, method_name, timestamp, state_json)"
            " VALUES ('flow', 'step', '2026-01-01T00:00:00+00:00', '{}')"
        )
        conn.commit()
    before = path.read_bytes()

    history = read_flow_state_history(path, "flow")

    assert _methods(history) == ["step"]
    assert path.read_bytes() == before
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='pending_feedback'"
            ).fetchone()[0]
            == 0
        )


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX file permissions are not enforced",
)
def test_reader_works_on_read_only_file(
    persistence: SQLiteFlowPersistence, db_path: str
) -> None:
    persistence.save_state("flow", "step", {})
    mode = os.stat(db_path).st_mode
    os.chmod(db_path, stat.S_IRUSR)
    try:
        assert _methods(read_flow_state_history(db_path, "flow")) == ["step"]
    finally:
        os.chmod(db_path, mode)


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX file permissions are not enforced",
)
def test_reader_works_when_directory_is_not_writable(tmp_path: Path) -> None:
    db_dir = tmp_path / "ro"
    db_dir.mkdir()
    path = db_dir / "flows.db"
    persistence = SQLiteFlowPersistence(str(path))
    persistence.save_state("flow", "step", {})
    # Make sure the WAL was checkpointed so only the main file remains.
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    for sidecar in (path.with_suffix(".db-wal"), path.with_suffix(".db-shm")):
        sidecar.unlink(missing_ok=True)
    dir_mode = db_dir.stat().st_mode
    db_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert _methods(read_flow_state_history(path, "flow")) == ["step"]
        assert sorted(p.name for p in db_dir.iterdir()) == ["flows.db"]
    finally:
        db_dir.chmod(dir_mode)


def test_base_persistence_does_not_support_history() -> None:
    class MemoryPersistence(FlowPersistence):
        def init_db(self) -> None:
            pass

        def save_state(
            self, flow_uuid: str, method_name: str, state_data: Any
        ) -> None:
            pass

        def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
            return None

    with pytest.raises(NotImplementedError):
        MemoryPersistence().list_state_history("flow")


def test_history_leaves_latest_state_and_pending_feedback_unchanged(
    persistence: SQLiteFlowPersistence,
) -> None:
    persistence.save_state("flow", "first", {"n": 1})
    context = PendingFeedbackContext(
        flow_id="flow",
        flow_class="tests.Flow",
        method_name="review",
        method_output="draft",
        message="Approve?",
    )
    persistence.save_pending_feedback("flow", context, {"n": 2})

    history = persistence.list_state_history("flow")

    assert _methods(history) == ["first", "review"]
    assert persistence.load_state("flow") == {"n": 2}
    pending = persistence.load_pending_feedback("flow")
    assert pending is not None
    assert pending[0] == {"n": 2}
    assert pending[1].method_name == "review"
