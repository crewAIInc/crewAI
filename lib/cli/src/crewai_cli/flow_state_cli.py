"""Read-only inspection of persisted Flow states.

Reads the ``flow_states`` table that ``SQLiteFlowPersistence`` writes: one row
per completed flow method. These rows are not the full-object checkpoints that
``crewai checkpoint`` browses.

Depends only on the standard library and click, so crewai-cli can read a store
without importing the full crewai framework. The database is opened read-only:
inspection never creates a missing file, writes state, or changes journal
settings.
"""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Any

import click

from crewai_cli.checkpoint_cli import _is_sqlite


STORAGE_KIND = "flow_states"

_SELECT_STATE = """
SELECT id, flow_uuid, method_name, timestamp, state_json
FROM flow_states
WHERE id = ? AND flow_uuid = ?
"""


def _read_only_uri(db_path: str) -> str:
    """Build a SQLite URI that reads the store without leaving files behind.

    The store is in WAL mode, where even a read-only connection creates
    ``-wal`` and ``-shm`` files next to the database. ``immutable=1`` avoids
    that, but it also skips a pending ``-wal``, so it is only used when no
    ``-wal`` file exists and every saved state is already in the main file.
    """
    path = Path(db_path).resolve()
    query = "mode=ro" if Path(f"{path}-wal").exists() else "mode=ro&immutable=1"
    return f"{path.as_uri()}?{query}"


def load_flow_state(
    db_path: str, flow_uuid: str, record_id: int
) -> dict[str, Any] | None:
    """Load one persisted state of a flow, scoped to that flow's UUID.

    Args:
        db_path: SQLite file written by ``SQLiteFlowPersistence``.
        flow_uuid: Unique identifier of the flow instance.
        record_id: Row id of the saved state in the ``flow_states`` table.
            This is not the flow's ``state.id``.

    Returns:
        The record with its storage kind, source, method name, timestamp and
        state, or None when the flow has no state with that row id.

    Raises:
        click.ClickException: If the file is missing, is not a SQLite database,
            has no readable ``flow_states`` table, or holds a state that is not
            a JSON object.
    """
    if not Path(db_path).is_file():
        raise click.ClickException(f"Flow state database not found: {db_path}")
    if not _is_sqlite(db_path):
        raise click.ClickException(f"Not a SQLite database: {db_path}")

    try:
        with closing(sqlite3.connect(_read_only_uri(db_path), uri=True)) as conn:
            row = conn.execute(_SELECT_STATE, (record_id, flow_uuid)).fetchone()
    except sqlite3.Error as exc:
        raise click.ClickException(
            f"Unsupported flow state database {db_path}: {exc}"
        ) from exc

    if row is None:
        return None

    row_id, row_uuid, method_name, timestamp, state_json = row
    try:
        state = json.loads(state_json)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(
            f"Record {record_id} does not hold valid JSON: {exc}"
        ) from exc
    if not isinstance(state, dict):
        raise click.ClickException(f"Record {record_id} does not hold a JSON object.")

    return {
        "storage": STORAGE_KIND,
        "db": db_path,
        "flow_uuid": row_uuid,
        "record_id": row_id,
        "method_name": method_name,
        "timestamp": timestamp,
        "state": state,
    }


def inspect_flow_state(
    db_path: str, flow_uuid: str, record_id: int, *, as_json: bool = False
) -> None:
    """Print one persisted state of a flow as text, or as JSON for scripts."""
    record = load_flow_state(db_path, flow_uuid, record_id)
    if record is None:
        raise click.ClickException(
            f"No state with record id {record_id} for flow {flow_uuid} in {db_path}"
        )

    if as_json:
        click.echo(json.dumps(record, indent=2))
        return

    click.echo("Storage:   flow state history (flow_states), not a checkpoint")
    click.echo(f"Source:    {record['db']}")
    click.echo(f"Flow UUID: {record['flow_uuid']}")
    click.echo(f"Record:    {record['record_id']}")
    click.echo(f"Method:    {record['method_name']}")
    click.echo(f"Time:      {record['timestamp']}")
    click.echo("State:")
    click.echo(json.dumps(record["state"], indent=2))
