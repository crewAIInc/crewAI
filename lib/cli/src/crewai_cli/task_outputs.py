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
            cursor = conn.cursor()
            cursor.execute("""
                SELECT *
                FROM latest_kickoff_task_outputs
                ORDER BY task_index
            """)
            rows = cursor.fetchall()
            results: list[dict[str, Any]] = [
                {
                    "task_id": row[0],
                    "expected_output": row[1],
                    "output": json.loads(row[2]),
                    "task_index": row[3],
                    "inputs": json.loads(row[4]),
                    "was_replayed": row[5],
                    "timestamp": row[6],
                }
                for row in rows
            ]
            return results
    except sqlite3.Error as e:
        logger.error("Failed to load task outputs: %s", e)
        return []
