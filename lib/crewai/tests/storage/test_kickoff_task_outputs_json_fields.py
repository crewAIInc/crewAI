"""Regression tests for JSON-backed task output fields (#7746)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from crewai.memory.storage.kickoff_task_outputs_storage import (
    KickoffTaskOutputsSQLiteStorage,
)
from crewai.task import Task


def _task() -> Task:
    return Task(description="Summarise the report", expected_output="A summary")


def _storage(tmp_path: Path) -> KickoffTaskOutputsSQLiteStorage:
    return KickoffTaskOutputsSQLiteStorage(db_path=str(tmp_path / "outputs.db"))


def test_update_serialises_non_dict_output(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    storage.add(_task(), {"raw": "done"}, task_index=0, inputs={"topic": "ai"})

    storage.update(0, output=["a", "b"])

    assert storage.load()[0]["output"] == ["a", "b"]


def test_load_handles_null_json_fields(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    storage.add(_task(), {"raw": "done"}, task_index=0, inputs={"topic": "ai"})

    storage.update(0, output=None, inputs=None)

    row = storage.load()[0]
    assert row["output"] is None
    assert row["inputs"] is None


def test_load_tolerates_legacy_raw_values(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    storage.add(_task(), {"raw": "done"}, task_index=0, inputs={"topic": "ai"})

    with sqlite3.connect(storage.db_path) as conn:
        conn.execute(
            "UPDATE latest_kickoff_task_outputs SET output = ?, inputs = ? "
            "WHERE task_index = 0",
            ("plain text", "also plain"),
        )

    row = storage.load()[0]
    assert row["output"] == "plain text"
    assert row["inputs"] == "also plain"
