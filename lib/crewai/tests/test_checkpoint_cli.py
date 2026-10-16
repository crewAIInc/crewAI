"""Tests for checkpoint CLI commands."""

from __future__ import annotations

from contextlib import closing
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from crewai_cli import checkpoint_cli
from crewai_cli.checkpoint_cli import (
    _info_json_file,
    _info_json_latest,
    _info_sqlite_id,
    _info_sqlite_latest,
    _list_json,
    _list_sqlite,
    _parse_checkpoint_json,
    _parse_duration,
    _prune_json,
    _prune_sqlite,
    _resolve_checkpoint,
    _task_list_from_meta,
    diff_checkpoints,
    prune_checkpoints,
    resume_checkpoint,
)
from crewai.state.provider.json_provider import JsonProvider


def _make_checkpoint_data(
    tasks_completed: int = 2,
    tasks_total: int = 4,
    trigger: str = "task_completed",
    branch: str = "main",
    parent_id: str | None = None,
    entity_type: str = "crew",
    name: str = "test_crew",
    inputs: dict[str, Any] | None = None,
) -> str:
    tasks: list[dict[str, Any]] = []
    for i in range(tasks_total):
        t: dict[str, Any] = {
            "description": f"Task {i + 1} description",
            "expected_output": f"Output {i + 1}",
        }
        if i < tasks_completed:
            t["output"] = {"raw": f"Result of task {i + 1}"}
        else:
            t["output"] = None
        tasks.append(t)

    data: dict[str, Any] = {
        "entities": [
            {
                "entity_type": entity_type,
                "name": name,
                "id": "abc12345-1234-1234-1234-abcdef012345",
                "tasks": tasks,
                "agents": [],
                "checkpoint_inputs": inputs or {},
            }
        ],
        "event_record": {"nodes": {f"node_{i}": {} for i in range(3)}},
        "trigger": trigger,
        "branch": branch,
        "parent_id": parent_id,
    }
    return json.dumps(data)


def _write_json_checkpoint(
    base_dir: str,
    branch: str = "main",
    name: str | None = None,
    data: str | None = None,
    tasks_completed: int = 2,
    inputs: dict[str, Any] | None = None,
) -> str:
    branch_dir = os.path.join(base_dir, branch)
    os.makedirs(branch_dir, exist_ok=True)
    if name is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        name = f"{ts}_abcd1234_p-none.json"
    path = os.path.join(branch_dir, name)
    if data is None:
        data = _make_checkpoint_data(tasks_completed=tasks_completed, inputs=inputs)
    with open(path, "w") as f:
        f.write(data)
    return path


def _create_sqlite_checkpoint(
    db_path: str,
    checkpoint_id: str | None = None,
    data: str | None = None,
    tasks_completed: int = 2,
    branch: str = "main",
    inputs: dict[str, Any] | None = None,
) -> str:
    if checkpoint_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        checkpoint_id = f"{ts}_abcd1234"
    if data is None:
        data = _make_checkpoint_data(
            tasks_completed=tasks_completed, branch=branch, inputs=inputs
        )
    # Close the setup handle too: a leaked one holds the database lock on Windows, which would
    # make the file-release assertions fail for the fixture's reason, not the command's.
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                parent_id TEXT,
                branch TEXT NOT NULL DEFAULT 'main',
                data JSONB NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO checkpoints (id, created_at, parent_id, branch, data) "
            "VALUES (?, ?, ?, ?, jsonb(?))",
            (checkpoint_id, checkpoint_id.split("_")[0], None, branch, data),
        )
        conn.commit()
    return checkpoint_id


class TestParseDuration:
    def test_days(self) -> None:
        assert _parse_duration("7d") == timedelta(days=7)

    def test_hours(self) -> None:
        assert _parse_duration("24h") == timedelta(hours=24)

    def test_minutes(self) -> None:
        assert _parse_duration("30m") == timedelta(minutes=30)

    def test_invalid_raises(self) -> None:
        with pytest.raises(Exception):
            _parse_duration("abc")

    def test_no_unit_raises(self) -> None:
        with pytest.raises(Exception):
            _parse_duration("7")


class TestResolveCheckpoint:
    def test_json_latest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_json_checkpoint(d, name="20260101T000000_aaaa1111_p-none.json")
            time.sleep(0.01)
            path2 = _write_json_checkpoint(
                d, name="20260102T000000_bbbb2222_p-none.json", tasks_completed=3
            )
            meta = _resolve_checkpoint(d, None)
            assert meta is not None
            assert meta["path"] == path2

    def test_json_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_json_checkpoint(d, name="20260101T000000_aaaa1111_p-none.json")
            _write_json_checkpoint(d, name="20260102T000000_bbbb2222_p-none.json")
            meta = _resolve_checkpoint(d, "aaaa1111")
            assert meta is not None
            assert "aaaa1111" in meta["name"]

    def test_json_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_json_checkpoint(d)
            assert _resolve_checkpoint(d, "nonexistent") is None

    def test_sqlite_latest(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            _create_sqlite_checkpoint(db_path, "20260101T000000_aaaa1111")
            _create_sqlite_checkpoint(
                db_path, "20260102T000000_bbbb2222", tasks_completed=3
            )
            meta = _resolve_checkpoint(db_path, None)
            assert meta is not None
            assert "bbbb2222" in meta["name"]

    def test_sqlite_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            _create_sqlite_checkpoint(db_path, "20260101T000000_aaaa1111")
            _create_sqlite_checkpoint(db_path, "20260102T000000_bbbb2222")
            meta = _resolve_checkpoint(db_path, "20260101T000000_aaaa1111")
            assert meta is not None
            assert "aaaa1111" in meta["name"]

    def test_sqlite_partial_id(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            _create_sqlite_checkpoint(db_path, "20260101T000000_aaaa1111")
            _create_sqlite_checkpoint(db_path, "20260102T000000_bbbb2222")
            meta = _resolve_checkpoint(db_path, "aaaa1111")
            assert meta is not None
            assert "aaaa1111" in meta["name"]

    def test_nonexistent(self) -> None:
        assert _resolve_checkpoint("/nonexistent/path", None) is None


class TestNonAsciiJsonCheckpoint:
    """JSON checkpoints are UTF-8 on disk; the CLI readers must decode them as such.

    ``JsonProvider`` writes checkpoints with ``encoding="utf-8"`` and the runtime
    serialises non-ASCII text verbatim, so readers that rely on the platform
    default encoding break on Windows (cp1252) for any non-ASCII checkpoint.
    """

    # "Đ" (U+0110) encodes to 0xC4 0x90; 0x90 is undefined in cp1252, so a
    # locale-dependent read fails loudly instead of silently producing mojibake.
    _NAME = "Đội ngũ phân tích"

    def _write_checkpoint(self, base_dir: str) -> str:
        """Write a UTF-8 checkpoint whose entity name is non-ASCII, as the runtime does."""
        data = json.loads(_make_checkpoint_data(name=self._NAME))
        raw = json.dumps(data, ensure_ascii=False)
        return JsonProvider().checkpoint(raw, base_dir, branch="main")

    def test_info_json_file_reads_utf8(self, tmp_path: Any) -> None:
        """``_info_json_file`` decodes a non-ASCII checkpoint on any platform."""
        path = self._write_checkpoint(str(tmp_path))
        meta = _info_json_file(path)
        assert meta["entities"][0]["name"] == self._NAME

    def test_info_json_latest_reads_utf8(self, tmp_path: Any) -> None:
        """``_info_json_latest`` decodes the newest non-ASCII checkpoint."""
        self._write_checkpoint(str(tmp_path))
        meta = _info_json_latest(str(tmp_path))
        assert meta is not None
        assert meta["entities"][0]["name"] == self._NAME

    def test_list_json_reads_utf8(self, tmp_path: Any) -> None:
        """``_list_json`` lists a non-ASCII checkpoint with its real size and entities."""
        self._write_checkpoint(str(tmp_path))
        results = _list_json(str(tmp_path))
        assert len(results) == 1
        assert results[0]["size"] > 0
        assert results[0]["entities"][0]["name"] == self._NAME


class TestTaskListFromMeta:
    def test_flattens_tasks(self) -> None:
        data = _make_checkpoint_data(tasks_completed=2, tasks_total=3)
        meta = _parse_checkpoint_json(data, "test")
        tasks = _task_list_from_meta(meta)
        assert len(tasks) == 3
        assert tasks[0]["completed"] is True
        assert tasks[2]["completed"] is False

    def test_empty_entities(self) -> None:
        assert _task_list_from_meta({"entities": []}) == []


class TestDiffCheckpoints:
    def test_diff_shows_status_change(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_json_checkpoint(
                d, name="20260101T000000_aaaa1111_p-none.json", tasks_completed=1
            )
            _write_json_checkpoint(
                d, name="20260102T000000_bbbb2222_p-none.json", tasks_completed=3
            )
            diff_checkpoints(d, "aaaa1111", "bbbb2222")
            out = capsys.readouterr().out
            assert "---" in out
            assert "+++" in out
            assert "status:" in out or "pending -> done" in out

    def test_diff_shows_output_change(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.TemporaryDirectory() as d:
            data1 = _make_checkpoint_data(tasks_completed=2)
            data2 = json.loads(data1)
            data2["entities"][0]["tasks"][0]["output"]["raw"] = "Updated result"
            _write_json_checkpoint(
                d,
                name="20260101T000000_aaaa1111_p-none.json",
                data=json.dumps(json.loads(data1)),
            )
            _write_json_checkpoint(
                d,
                name="20260102T000000_bbbb2222_p-none.json",
                data=json.dumps(data2),
            )
            diff_checkpoints(d, "aaaa1111", "bbbb2222")
            out = capsys.readouterr().out
            assert "output:" in out

    def test_diff_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_json_checkpoint(d, name="20260101T000000_aaaa1111_p-none.json")
            diff_checkpoints(d, "aaaa1111", "nonexistent")
            out = capsys.readouterr().out
            assert "not found" in out

    def test_diff_input_change(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_json_checkpoint(
                d,
                name="20260101T000000_aaaa1111_p-none.json",
                inputs={"topic": "AI"},
            )
            _write_json_checkpoint(
                d,
                name="20260102T000000_bbbb2222_p-none.json",
                inputs={"topic": "ML"},
            )
            diff_checkpoints(d, "aaaa1111", "bbbb2222")
            out = capsys.readouterr().out
            assert "Inputs:" in out
            assert "AI" in out
            assert "ML" in out


class TestPruneJson:
    def test_keep_n(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for i in range(5):
                _write_json_checkpoint(
                    d, name=f"2026010{i + 1}T000000_aaa{i}1111_p-none.json"
                )
                time.sleep(0.01)
            deleted = _prune_json(d, keep=2, older_than=None)
            assert deleted == 3
            remaining = []
            for root, _, files in os.walk(d):
                remaining.extend(files)
            assert len(remaining) == 2

    def test_older_than(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            old_path = _write_json_checkpoint(
                d, name="20250101T000000_old01111_p-none.json"
            )
            os.utime(old_path, (0, 0))
            _write_json_checkpoint(d, name="20990101T000000_new01111_p-none.json")
            deleted = _prune_json(d, keep=None, older_than=timedelta(days=1))
            assert deleted == 1

    def test_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            assert _prune_json(d, keep=2, older_than=None) == 0

    def test_removes_empty_branch_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = _write_json_checkpoint(
                d,
                branch="feature",
                name="20260101T000000_aaaa1111_p-none.json",
            )
            os.utime(path, (0, 0))
            _prune_json(d, keep=None, older_than=timedelta(days=1))
            assert not os.path.exists(os.path.join(d, "feature"))


class TestPruneSqlite:
    def test_keep_n(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            for i in range(5):
                _create_sqlite_checkpoint(
                    db_path, f"2026010{i + 1}T000000_aaa{i}1111"
                )
            deleted = _prune_sqlite(db_path, keep=2, older_than=None)
            assert deleted == 3
            with closing(sqlite3.connect(db_path)) as conn, conn:
                count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            assert count == 2

    def test_older_than(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            _create_sqlite_checkpoint(db_path, "20200101T000000_old01111")
            _create_sqlite_checkpoint(db_path, "20990101T000000_new01111")
            deleted = _prune_sqlite(db_path, keep=None, older_than=timedelta(days=1))
            assert deleted >= 1
            with closing(sqlite3.connect(db_path)) as conn, conn:
                count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            assert count >= 1


class TestPruneCommand:
    def test_no_options_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.TemporaryDirectory() as d:
            prune_checkpoints(d, keep=None, older_than=None)
            out = capsys.readouterr().out
            assert "Specify" in out

    def test_dry_run_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_json_checkpoint(d)
            prune_checkpoints(d, keep=1, older_than=None, dry_run=True)
            out = capsys.readouterr().out
            assert "Would prune" in out

    def test_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        prune_checkpoints("/nonexistent", keep=1, older_than=None)
        out = capsys.readouterr().out
        assert "Not a directory" in out


class TestResumeCheckpoint:
    def test_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.TemporaryDirectory() as d:
            resume_checkpoint(d, "nonexistent")
            out = capsys.readouterr().out
            assert "not found" in out

    def test_no_checkpoints(self, capsys: pytest.CaptureFixture[str]) -> None:
        with tempfile.TemporaryDirectory() as d:
            resume_checkpoint(d, None)
            out = capsys.readouterr().out
            assert "No checkpoints" in out


class TestDiscoverabilityMessage:
    def test_checkpoint_listener_logs_resume_hint(self) -> None:
        from crewai.state.checkpoint_listener import _do_checkpoint
        from crewai.state.runtime import RuntimeState

        state = MagicMock(spec=RuntimeState)
        state.root = []
        state.model_dump.return_value = {"entities": [], "event_record": {"nodes": {}}}
        state._parent_id = None
        state._branch = "main"

        cfg = MagicMock()
        cfg.location = "/tmp/cp"
        cfg.max_checkpoints = None
        cfg.provider.checkpoint.return_value = "/tmp/cp/main/20260101T000000_test1234_p-none.json"
        cfg.provider.extract_id.return_value = "20260101T000000_test1234"

        with (
            patch("crewai.state.checkpoint_listener._prepare_entities"),
            patch("crewai.state.checkpoint_listener.logger") as mock_logger,
        ):
            _do_checkpoint(state, cfg)

        cfg.provider.extract_id.assert_called_once()
        mock_logger.info.assert_called_once()
        logged: str = mock_logger.info.call_args[0][0]
        assert "crewai checkpoint resume" in logged
        assert "20260101T000000_test1234" in logged


class TestSqliteConnectionLifecycle:
    """Every checkpoint SQLite reader/writer closes the connection it opened.

    ``with sqlite3.connect(...) as conn`` only commits or rolls back; it never
    closes. The connection then survives in a reference cycle (the statement
    cache) until a cyclic GC pass, so the OS file handle outlives the call and
    on Windows keeps ``flow_states.db``/the checkpoint database locked.
    """

    @staticmethod
    def _track_connections(monkeypatch: pytest.MonkeyPatch) -> list[sqlite3.Connection]:
        """Record every connection opened through the module under test."""
        opened: list[sqlite3.Connection] = []
        real_connect = sqlite3.connect

        def tracking_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        monkeypatch.setattr(checkpoint_cli.sqlite3, "connect", tracking_connect)
        return opened

    @staticmethod
    def _assert_all_closed(opened: list[sqlite3.Connection]) -> None:
        assert opened, "expected the command to open a connection"
        for conn in opened:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                conn.execute("SELECT 1")

    def test_list_sqlite_closes_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            _create_sqlite_checkpoint(db_path, "20260101T000000_aaaa1111")

            opened = self._track_connections(monkeypatch)
            entries = _list_sqlite(db_path)

            assert len(entries) == 1
            self._assert_all_closed(opened)

    def test_info_sqlite_closes_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            _create_sqlite_checkpoint(db_path, "20260101T000000_aaaa1111")

            opened = self._track_connections(monkeypatch)
            assert _info_sqlite_latest(db_path) is not None
            assert _info_sqlite_id(db_path, "20260101T000000_aaaa1111") is not None

            self._assert_all_closed(opened)

    def test_prune_sqlite_closes_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            for i in range(3):
                _create_sqlite_checkpoint(db_path, f"2026010{i + 1}T000000_aaa{i}1111")

            opened = self._track_connections(monkeypatch)
            assert _prune_sqlite(db_path, keep=1, older_than=None) == 2

            self._assert_all_closed(opened)

    def test_prune_dry_run_closes_connection(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            _create_sqlite_checkpoint(db_path, "20260101T000000_aaaa1111")

            opened = self._track_connections(monkeypatch)
            prune_checkpoints(db_path, keep=1, older_than=None, dry_run=True)

            assert "Would prune" in capsys.readouterr().out
            self._assert_all_closed(opened)

    def test_database_file_is_releasable_after_use(self) -> None:
        """The database can be removed straight away, without a GC pass.

        This is the user-visible symptom on Windows, where a lingering handle
        blocks ``os.remove``/``os.replace`` with ``PermissionError``.
        """
        with tempfile.TemporaryDirectory() as d:
            db_path = os.path.join(d, "test.db")
            _create_sqlite_checkpoint(db_path, "20260101T000000_aaaa1111")

            _list_sqlite(db_path)

            os.remove(db_path)
            assert not os.path.exists(db_path)
