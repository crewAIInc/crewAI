"""`crewai flow state diff` compares two saved states of one flow, read-only."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from click.testing import CliRunner
from crewai.flow.persistence import SQLiteFlowPersistence
import pytest

from crewai_cli.cli import crewai


def _row_ids(db_path: str, flow_uuid: str) -> list[int]:
    """Row ids saved for a flow, oldest first."""
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id FROM flow_states WHERE flow_uuid = ? ORDER BY id",
            (flow_uuid,),
        ).fetchall()
    return [row[0] for row in rows]


def _invoke(*args: str):
    return CliRunner().invoke(crewai, ["flow", "state", "diff", *args])


@pytest.fixture
def flow_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "flow_states.db")
    persistence = SQLiteFlowPersistence(db_path)
    persistence.save_state(
        "flow-a",
        "begin",
        {"id": "flow-a", "counter": 1, "draft": None, "nested": {"items": [1], "ok": True}},
    )
    persistence.save_state("flow-b", "begin", {"id": "flow-b", "counter": 1})
    persistence.save_state(
        "flow-a",
        "finish",
        {
            "id": "flow-a",
            "counter": 2,
            "nested": {"items": [1, 2], "ok": True},
            "result": "done",
        },
    )
    return db_path


def test_diff_prints_both_records_and_one_line_per_change(flow_db: str) -> None:
    first, second = _row_ids(flow_db, "flow-a")

    result = _invoke("--db", flow_db, "flow-a", str(first), str(second))

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith(f"--- record {first}  begin  ")
    assert lines[1].startswith(f"+++ record {second}  finish  ")
    assert lines[2] == ""
    assert lines[3:] == [
        "~ counter: 1 -> 2",
        "- draft: null",
        "~ nested.items: [1] -> [1, 2]",
        '+ result: "done"',
    ]


def test_diff_json_output_carries_both_records_and_the_changes(flow_db: str) -> None:
    first, second = _row_ids(flow_db, "flow-a")

    result = _invoke("--db", flow_db, "flow-a", str(first), str(second), "--json")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["flow_uuid"] == "flow-a"
    assert payload["before"]["id"] == first
    assert payload["before"]["method_name"] == "begin"
    assert payload["after"]["id"] == second
    assert payload["after"]["method_name"] == "finish"
    assert payload["identical"] is False
    assert [change["path"] for change in payload["changes"]] == [
        ["counter"],
        ["draft"],
        ["nested", "items"],
        ["result"],
    ]
    assert payload["changes"][1] == {"path": ["draft"], "kind": "removed", "before": None}
    assert payload["changes"][3] == {"path": ["result"], "kind": "added", "after": "done"}


def test_comparing_a_record_with_itself_reports_no_differences(flow_db: str) -> None:
    first, _ = _row_ids(flow_db, "flow-a")

    text = _invoke("--db", flow_db, "flow-a", str(first), str(first))
    as_json = _invoke("--db", flow_db, "flow-a", str(first), str(first), "--json")

    assert text.exit_code == 0, text.output
    assert text.output.rstrip().endswith("No differences.")
    assert as_json.exit_code == 0, as_json.output
    assert json.loads(as_json.output) == {
        "flow_uuid": "flow-a",
        "before": json.loads(as_json.output)["before"],
        "after": json.loads(as_json.output)["before"],
        "identical": True,
        "changes": [],
    }


def test_a_record_saved_by_another_flow_is_reported_as_missing(flow_db: str) -> None:
    first, _ = _row_ids(flow_db, "flow-a")
    (other,) = _row_ids(flow_db, "flow-b")

    result = _invoke("--db", flow_db, "flow-a", str(first), str(other))

    assert result.exit_code == 1
    assert f"Flow flow-a has no state record {other}" in result.output


def test_an_unknown_record_id_is_an_error(flow_db: str) -> None:
    ids = _row_ids(flow_db, "flow-a")
    unknown = max(ids) + 100

    result = _invoke("--db", flow_db, "flow-a", str(unknown), str(ids[0]))

    assert result.exit_code == 1
    assert f"Flow flow-a has no state record {unknown}" in result.output


def test_a_missing_database_is_an_error_and_is_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"

    result = _invoke("--db", str(missing), "flow-a", "1", "2")

    assert result.exit_code == 1
    assert "Flow state database not found" in result.output
    assert not missing.exists()


def test_record_ids_must_be_integers(flow_db: str) -> None:
    result = _invoke("--db", flow_db, "flow-a", "first", "2")

    assert result.exit_code == 2
    assert "Invalid value for 'BEFORE_ID'" in result.output


def test_default_database_is_the_one_persistence_writes_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("crewai_core.paths.db_storage_path", lambda: str(tmp_path))
    persistence = SQLiteFlowPersistence(str(tmp_path / "flow_states.db"))
    persistence.save_state("flow-a", "begin", {"n": 1})
    persistence.save_state("flow-a", "finish", {"n": 2})
    first, second = _row_ids(str(tmp_path / "flow_states.db"), "flow-a")

    result = _invoke("flow-a", str(first), str(second))

    assert result.exit_code == 0, result.output
    assert "~ n: 1 -> 2" in result.output
