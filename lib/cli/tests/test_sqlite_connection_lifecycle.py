import sqlite3
from unittest.mock import patch

import pytest

from crewai_cli import checkpoint_cli
from crewai_cli.checkpoint_cli import (
    _info_sqlite_id,
    _info_sqlite_latest,
    _list_sqlite,
    _prune_sqlite,
    prune_checkpoints,
)
from crewai_cli.task_outputs import load_task_outputs


def _create_checkpoint_db(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE checkpoints "
            "(id TEXT PRIMARY KEY, created_at TEXT, data TEXT)"
        )
        conn.executemany(
            "INSERT INTO checkpoints (id, created_at, data) VALUES (?, ?, ?)",
            [
                ("checkpoint-1", "20260101T000000", "{}"),
                ("checkpoint-2", "20260102T000000", "{}"),
            ],
        )


def _create_task_outputs_db(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE latest_kickoff_task_outputs ("
            "task_id TEXT PRIMARY KEY, task_key TEXT, expected_output TEXT, "
            "output JSON, task_index INTEGER, inputs JSON, "
            "was_replayed BOOLEAN, timestamp DATETIME)"
        )
        conn.execute(
            "INSERT INTO latest_kickoff_task_outputs "
            "(task_id, expected_output, output, task_index, inputs, "
            "was_replayed, timestamp) "
            "VALUES ('task-1', 'done', '{}', 0, '{}', 0, '2026-01-01')"
        )


def _track_connections(opened):
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    return connect


def _assert_connections_closed(opened):
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            conn.execute("SELECT 1")


def _run_checkpoint_operation(operation, db_path):
    if operation == "list":
        _list_sqlite(str(db_path))
    elif operation == "latest":
        _info_sqlite_latest(str(db_path))
    elif operation == "by_id":
        _info_sqlite_id(str(db_path), "checkpoint-1")
    elif operation == "prune":
        assert _prune_sqlite(str(db_path), keep=1, older_than=None) == 1
    else:
        prune_checkpoints(str(db_path), keep=1, older_than=None, dry_run=True)


@pytest.mark.parametrize("operation", ["list", "latest", "by_id", "prune", "prune_dry_run"])
def test_checkpoint_operations_close_connections(tmp_path, operation):
    db_path = tmp_path / "checkpoints.db"
    _create_checkpoint_db(db_path)
    opened = []

    with patch.object(
        checkpoint_cli,
        "_record_checkpoint_usage",
    ), patch.object(
        checkpoint_cli.sqlite3,
        "connect",
        _track_connections(opened),
    ):
        _run_checkpoint_operation(operation, db_path)

    _assert_connections_closed(opened)


def test_prune_sqlite_commits_deleted_rows(tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _create_checkpoint_db(db_path)

    assert _prune_sqlite(str(db_path), keep=1, older_than=None) == 1

    with sqlite3.connect(db_path) as conn:
        remaining = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]

    assert remaining == 1


def test_load_task_outputs_close_connections(tmp_path):
    db_path = tmp_path / "task_outputs.db"
    _create_task_outputs_db(db_path)
    opened = []

    with patch.object(
        checkpoint_cli.sqlite3,
        "connect",
        _track_connections(opened),
    ):
        outputs = load_task_outputs(str(db_path))

    assert len(outputs) == 1
    _assert_connections_closed(opened)


def test_load_task_outputs_close_connections_on_query_error(tmp_path):
    db_path = tmp_path / "task_outputs.db"
    db_path.touch()
    opened = []

    with patch.object(
        checkpoint_cli.sqlite3,
        "connect",
        _track_connections(opened),
    ):
        assert load_task_outputs(str(db_path)) == []

    _assert_connections_closed(opened)
