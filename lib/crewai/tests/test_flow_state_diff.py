"""Tests for comparing two persisted flow states and reading them back."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest
from crewai.flow.persistence import (
    FlowStateChange,
    SQLiteFlowPersistence,
    compare_flow_states,
    read_flow_state,
)


def _changes(before: dict[Any, Any], after: dict[Any, Any]) -> list[dict[str, Any]]:
    return compare_flow_states(before, after).to_dict()["changes"]


def _row_ids(db_path: str | Path, flow_uuid: str) -> list[int]:
    """Row ids saved for a flow, oldest first."""
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id FROM flow_states WHERE flow_uuid = ? ORDER BY id",
            (flow_uuid,),
        ).fetchall()
    return [row[0] for row in rows]


# compare_flow_states


def test_identical_states_have_no_changes() -> None:
    state = {"id": "f", "counter": 1, "nested": {"items": [1, 2], "flag": None}}

    diff = compare_flow_states(state, json.loads(json.dumps(state)))

    assert diff.identical
    assert diff.changes == ()
    assert diff.to_dict() == {"identical": True, "changes": []}
    assert diff.render_text() == "No differences."


def test_added_and_removed_keys_carry_only_the_side_that_had_them() -> None:
    diff = compare_flow_states(
        {"id": "f", "draft": "v1"}, {"id": "f", "result": "done"}
    )

    assert not diff.identical
    assert diff.to_dict() == {
        "identical": False,
        "changes": [
            {"path": ["draft"], "kind": "removed", "before": "v1"},
            {"path": ["result"], "kind": "added", "after": "done"},
        ],
    }


def test_nested_changes_carry_the_full_path() -> None:
    before = {"a": {"b": {"c": 1, "keep": True}, "x": 1}}
    after = {"a": {"b": {"c": 2, "keep": True}, "x": 1}}

    assert _changes(before, after) == [
        {"path": ["a", "b", "c"], "kind": "changed", "before": 1, "after": 2}
    ]


def test_nested_additions_and_removals_report_the_subtree() -> None:
    before = {"a": {"gone": {"deep": 1}, "same": 0}}
    after = {"a": {"new": {"deep": 2}, "same": 0}}

    assert _changes(before, after) == [
        {"path": ["a", "gone"], "kind": "removed", "before": {"deep": 1}},
        {"path": ["a", "new"], "kind": "added", "after": {"deep": 2}},
    ]


def test_lists_are_compared_as_whole_values() -> None:
    before = {"items": [1, 2, 3], "rows": [{"k": 1}], "same": ["a"]}
    after = {"items": [1, 2, 4], "rows": [{"k": 2}], "same": ["a"]}

    assert _changes(before, after) == [
        {
            "path": ["items"],
            "kind": "changed",
            "before": [1, 2, 3],
            "after": [1, 2, 4],
        },
        {"path": ["rows"], "kind": "changed", "before": [{"k": 1}], "after": [{"k": 2}]},
    ]


def test_absent_key_differs_from_key_set_to_none() -> None:
    assert _changes({"a": None}, {}) == [
        {"path": ["a"], "kind": "removed", "before": None}
    ]
    assert _changes({}, {"a": None}) == [
        {"path": ["a"], "kind": "added", "after": None}
    ]
    assert _changes({"a": None}, {"a": None}) == []
    assert _changes({"a": None}, {"a": 0}) == [
        {"path": ["a"], "kind": "changed", "before": None, "after": 0}
    ]


def test_values_compare_by_json_type_as_well_as_value() -> None:
    assert _changes({"a": 1}, {"a": 1}) == []
    assert _changes({"a": 1}, {"a": True}) == [
        {"path": ["a"], "kind": "changed", "before": 1, "after": True}
    ]
    assert _changes({"a": 1}, {"a": 1.0}) == [
        {"path": ["a"], "kind": "changed", "before": 1, "after": 1.0}
    ]
    assert _changes({"a": "1"}, {"a": 1}) == [
        {"path": ["a"], "kind": "changed", "before": "1", "after": 1}
    ]


def test_dict_replaced_by_another_type_is_one_change() -> None:
    assert _changes({"a": {"b": 1}}, {"a": [1]}) == [
        {"path": ["a"], "kind": "changed", "before": {"b": 1}, "after": [1]}
    ]
    assert _changes({"a": 1}, {"a": {"b": 1}}) == [
        {"path": ["a"], "kind": "changed", "before": 1, "after": {"b": 1}}
    ]


def test_changes_are_sorted_by_path_whatever_the_key_order() -> None:
    before = {"z": 1, "m": {"b": 1, "a": 1}, "a": 1}
    after = {"a": 2, "m": {"a": 2, "b": 2}, "z": 2}

    diff = compare_flow_states(before, after)

    assert [change.path for change in diff.changes] == [
        ("a",),
        ("m", "a"),
        ("m", "b"),
        ("z",),
    ]
    shuffled = dict(reversed(list(before.items())))
    assert compare_flow_states(shuffled, after) == diff


def test_non_string_keys_are_compared_by_their_text() -> None:
    assert _changes({1: "a", "b": 1}, {1: "b", "b": 1}) == [
        {"path": ["1"], "kind": "changed", "before": "a", "after": "b"}
    ]


def test_to_dict_round_trips_through_json() -> None:
    diff = compare_flow_states(
        {"a": {"b": [1, None]}}, {"a": {"b": [1]}, "c": None}
    )

    assert json.loads(json.dumps(diff.to_dict())) == diff.to_dict()


def test_render_text_marks_each_kind_and_keeps_values_short() -> None:
    diff = compare_flow_states(
        {"removed": None, "changed": "old", "long": "x" * 200},
        {"added": [1, 2], "changed": "new", "long": "y" * 200},
    )

    lines = diff.render_text().splitlines()

    assert lines[0] == "+ added: [1, 2]"
    assert lines[1] == '~ changed: "old" -> "new"'
    assert lines[2].startswith('~ long: "xxx')
    before_preview, after_preview = lines[2].removeprefix("~ long: ").split(" -> ")
    assert before_preview.endswith("...") and len(before_preview) == 80
    assert after_preview.endswith("...") and len(after_preview) == 80
    assert lines[3] == "- removed: null"


def test_change_exposes_a_dotted_path() -> None:
    assert FlowStateChange(("a", "b"), "added", after=1).dotted_path == "a.b"


# read_flow_state


@pytest.fixture
def populated_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "flows.db")
    persistence = SQLiteFlowPersistence(db_path)
    persistence.save_state(
        "flow-a", "first", {"id": "flow-a", "step": 1, "nested": {"ok": True}}
    )
    persistence.save_state("flow-b", "other", {"id": "flow-b", "step": 1})
    persistence.save_state(
        "flow-a", "second", {"id": "flow-a", "step": 2, "nested": {"ok": False}}
    )
    return db_path


def test_read_flow_state_returns_the_record_with_its_metadata(
    populated_db: str,
) -> None:
    first_id, second_id = _row_ids(populated_db, "flow-a")

    first = read_flow_state(populated_db, "flow-a", first_id)
    second = read_flow_state(populated_db, "flow-a", second_id)

    assert first is not None
    assert first.id == first_id
    assert first.method_name == "first"
    assert first.timestamp.tzinfo is not None
    assert first.state == {"id": "flow-a", "step": 1, "nested": {"ok": True}}
    assert second is not None
    assert second.method_name == "second"
    assert second.state["step"] == 2
    assert second.timestamp >= first.timestamp


def test_read_flow_state_never_returns_another_flows_record(
    populated_db: str,
) -> None:
    (other_id,) = _row_ids(populated_db, "flow-b")

    assert read_flow_state(populated_db, "flow-a", other_id) is None
    assert read_flow_state(populated_db, "flow-b", other_id) is not None


def test_read_flow_state_returns_none_for_unknown_id_or_flow(
    populated_db: str,
) -> None:
    ids = _row_ids(populated_db, "flow-a")

    assert read_flow_state(populated_db, "flow-a", max(ids) + 100) is None
    assert read_flow_state(populated_db, "unknown-flow", ids[0]) is None


def test_read_flow_state_does_not_create_a_missing_database(tmp_path: Path) -> None:
    missing = tmp_path / "missing.db"

    assert read_flow_state(missing, "flow", 1) is None
    assert read_flow_state(str(missing), "flow", 1) is None
    assert read_flow_state(tmp_path, "flow", 1) is None

    assert not missing.exists()


def test_read_flow_state_returns_none_when_table_is_missing(tmp_path: Path) -> None:
    path = tmp_path / "other.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
        conn.commit()

    assert read_flow_state(path, "flow", 1) is None

    with closing(sqlite3.connect(path)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables == {"unrelated"}


def _write_plain_database(path: Path, state_json: str) -> None:
    """A flow_states table written without SQLiteFlowPersistence, in DELETE journal mode."""
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE flow_states (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " flow_uuid TEXT NOT NULL, method_name TEXT NOT NULL,"
            " timestamp DATETIME NOT NULL, state_json TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO flow_states (flow_uuid, method_name, timestamp, state_json)"
            " VALUES ('flow', 'step', '2026-01-01T00:00:00', ?)",
            (state_json,),
        )
        conn.commit()


def test_read_flow_state_leaves_the_database_untouched(tmp_path: Path) -> None:
    path = tmp_path / "plain.db"
    _write_plain_database(path, '{"n": 1}')
    before = path.read_bytes()

    record = read_flow_state(path, "flow", 1)

    assert record is not None
    assert record.state == {"n": 1}
    assert record.timestamp == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["plain.db"]
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name='pending_feedback'"
            ).fetchone()[0]
            == 0
        )


def test_read_flow_state_returns_none_for_a_non_object_state(tmp_path: Path) -> None:
    path = tmp_path / "plain.db"
    _write_plain_database(path, "[1, 2]")

    assert read_flow_state(path, "flow", 1) is None
