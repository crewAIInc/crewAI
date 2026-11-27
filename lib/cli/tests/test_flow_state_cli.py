from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from click.testing import CliRunner
import pytest

from crewai_cli.cli import flow_state


FLOW_UUID = "11111111-1111-4111-8111-111111111111"
OTHER_UUID = "22222222-2222-4222-8222-222222222222"

_SCHEMA = """
CREATE TABLE flow_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_uuid TEXT NOT NULL,
    method_name TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    state_json TEXT NOT NULL
)
"""


def _create_store(
    path: Path, rows: list[tuple[str, str, str, str]] | None = None
) -> Path:
    """Write a flow_states store the way SQLiteFlowPersistence lays it out."""
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_SCHEMA)
        conn.executemany(
            "INSERT INTO flow_states (flow_uuid, method_name, timestamp, state_json) "
            "VALUES (?, ?, ?, ?)",
            rows or [],
        )
    return path


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return _create_store(
        tmp_path / "flow_states.db",
        [
            (
                FLOW_UUID,
                "draft",
                "2026-11-15T10:00:00+00:00",
                json.dumps({"id": FLOW_UUID, "topic": "café", "count": 1}),
            ),
            (
                OTHER_UUID,
                "review",
                "2026-11-15T10:05:00+00:00",
                json.dumps({"id": OTHER_UUID, "approved": None}),
            ),
        ],
    )


def _invoke(*args: str) -> tuple[int, str]:
    result = CliRunner().invoke(flow_state, list(args))
    return result.exit_code, result.output


def test_json_output_is_the_stored_record(store: Path) -> None:
    exit_code, output = _invoke(str(store), FLOW_UUID, "1", "--json")

    assert exit_code == 0
    assert json.loads(output) == {
        "storage": "flow_states",
        "db": str(store),
        "flow_uuid": FLOW_UUID,
        "record_id": 1,
        "method_name": "draft",
        "timestamp": "2026-11-15T10:00:00+00:00",
        "state": {"id": FLOW_UUID, "topic": "café", "count": 1},
    }


def test_text_output_names_the_storage_kind_and_the_record(store: Path) -> None:
    exit_code, output = _invoke(str(store), OTHER_UUID, "2")

    assert exit_code == 0
    assert "flow state history (flow_states), not a checkpoint" in output
    assert f"Source:    {store}" in output
    assert f"Flow UUID: {OTHER_UUID}" in output
    assert "Record:    2" in output
    assert "Method:    review" in output
    assert "Time:      2026-11-15T10:05:00+00:00" in output
    assert '"approved": null' in output


def test_record_of_another_flow_is_not_shown(store: Path) -> None:
    exit_code, output = _invoke(str(store), FLOW_UUID, "2")

    assert exit_code == 1
    assert f"No state with record id 2 for flow {FLOW_UUID}" in output
    assert "approved" not in output


def test_unknown_record_id_fails(store: Path) -> None:
    exit_code, output = _invoke(str(store), FLOW_UUID, "99", "--json")

    assert exit_code == 1
    assert "No state with record id 99" in output


def test_empty_store_reports_no_state(tmp_path: Path) -> None:
    store = _create_store(tmp_path / "empty.db")

    exit_code, output = _invoke(str(store), FLOW_UUID, "1")

    assert exit_code == 1
    assert "No state with record id 1" in output


def test_missing_database_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"

    exit_code, output = _invoke(str(missing), FLOW_UUID, "1")

    assert exit_code == 1
    assert f"Flow state database not found: {missing}" in output
    assert list(tmp_path.iterdir()) == []


def test_file_that_is_not_sqlite_is_rejected(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.db"
    text_file.write_text("not a database", encoding="utf-8")

    exit_code, output = _invoke(str(text_file), FLOW_UUID, "1")

    assert exit_code == 1
    assert f"Not a SQLite database: {text_file}" in output


def test_sqlite_file_without_flow_states_is_unsupported(tmp_path: Path) -> None:
    other = tmp_path / "checkpoints.db"
    with closing(sqlite3.connect(other)) as conn, conn:
        conn.execute("CREATE TABLE checkpoints (id TEXT PRIMARY KEY, data BLOB)")

    exit_code, output = _invoke(str(other), FLOW_UUID, "1")

    assert exit_code == 1
    assert f"Unsupported flow state database {other}" in output
    assert "flow_states" in output


@pytest.mark.parametrize("state_json", ["[1, 2]", '"text"', "{broken"])
def test_state_that_is_not_a_json_object_is_rejected(
    tmp_path: Path, state_json: str
) -> None:
    store = _create_store(
        tmp_path / "flow_states.db",
        [(FLOW_UUID, "draft", "2026-11-15T10:00:00+00:00", state_json)],
    )

    exit_code, output = _invoke(str(store), FLOW_UUID, "1", "--json")

    assert exit_code == 1
    assert "Record 1 does not hold" in output


@pytest.mark.parametrize("record_id", ["0", "-1", "abc", "1.5"])
def test_invalid_record_id_is_a_usage_error(store: Path, record_id: str) -> None:
    exit_code, output = _invoke(str(store), FLOW_UUID, "--", record_id)

    assert exit_code == 2
    assert "RECORD_ID" in output


def test_inspection_leaves_the_store_untouched(store: Path) -> None:
    before_bytes = store.read_bytes()
    before_files = sorted(path.name for path in store.parent.iterdir())

    for args in (
        (str(store), FLOW_UUID, "1"),
        (str(store), FLOW_UUID, "1", "--json"),
        (str(store), FLOW_UUID, "99"),
    ):
        _invoke(*args)

    assert store.read_bytes() == before_bytes
    assert sorted(path.name for path in store.parent.iterdir()) == before_files
    with closing(sqlite3.connect(store)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_state_still_in_the_write_ahead_log_is_shown(store: Path) -> None:
    """A flow that is still running keeps its newest states in the -wal file."""
    with closing(sqlite3.connect(store)) as writer:
        writer.execute(
            "INSERT INTO flow_states (flow_uuid, method_name, timestamp, state_json) "
            "VALUES (?, ?, ?, ?)",
            (FLOW_UUID, "publish", "2026-11-15T10:10:00+00:00", '{"count": 2}'),
        )
        writer.commit()
        before_files = sorted(path.name for path in store.parent.iterdir())

        exit_code, output = _invoke(str(store), FLOW_UUID, "3", "--json")

        assert before_files == [store.name, f"{store.name}-shm", f"{store.name}-wal"]
        assert sorted(path.name for path in store.parent.iterdir()) == before_files
    assert exit_code == 0
    assert json.loads(output)["state"] == {"count": 2}


def test_reads_a_store_written_by_sqlite_flow_persistence(tmp_path: Path) -> None:
    sqlite_persistence = pytest.importorskip("crewai.flow.persistence.sqlite")
    db_path = tmp_path / "flow_states.db"
    persistence = sqlite_persistence.SQLiteFlowPersistence(str(db_path))
    persistence.save_state(FLOW_UUID, "draft", {"id": FLOW_UUID, "count": 1})
    persistence.save_state(FLOW_UUID, "review", {"id": FLOW_UUID, "count": 2})

    exit_code, output = _invoke(str(db_path), FLOW_UUID, "2", "--json")

    assert exit_code == 0
    record = json.loads(output)
    assert record["method_name"] == "review"
    assert record["state"] == persistence.load_state_from_history(FLOW_UUID, 2)
