"""Lightweight SQLite reader for kickoff task outputs.

Only used by the ``crewai log-tasks-outputs`` CLI command.  Depends solely on
the standard library + *appdirs* so crewai-cli can read stored outputs without
importing the full crewai framework.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

from crewai_cli.user_data import _db_storage_path


logger = logging.getLogger(__name__)


def load_task_outputs(db_path: str | None = None) -> list[dict[str, Any]]:
    """Return all rows from the kickoff task outputs database."""
    if db_path is None:
        db_path = str(Path(_db_storage_path()) / "latest_kickoff_task_outputs.db")

    if not Path(db_path).exists():
        return []

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT task_id, expected_output, output, task_index,
                       inputs, was_replayed, timestamp
                FROM latest_kickoff_task_outputs
                ORDER BY task_index
            """)
            rows = cursor.fetchall()
    except sqlite3.Error as e:
        logger.error("Failed to load task outputs: %s", e)
        return []

    return [
        {
            "task_id": row["task_id"],
            "expected_output": row["expected_output"],
            "output": _safe_json_loads(row["output"]),
            "task_index": row["task_index"],
            "inputs": _safe_json_loads(row["inputs"]),
            "was_replayed": row["was_replayed"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]


def _safe_json_loads(value: str | None) -> Any:
    """Decode a JSON column tolerantly: NULL/blank/corrupt → None."""
    if not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning("Failed to decode JSON column: %s", e)
        return None
