"""CLI commands for inspecting states saved by ``SQLiteFlowPersistence``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click


if TYPE_CHECKING:
    from crewai.flow.persistence import FlowStateRecord


def default_flow_state_db() -> str:
    """The database ``SQLiteFlowPersistence`` writes to when given no path."""
    from crewai_core.paths import db_storage_path

    return str(Path(db_storage_path()) / "flow_states.db")


def _record_summary(record: FlowStateRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "method_name": record.method_name,
        "timestamp": record.timestamp.isoformat(),
    }


def _record_header(marker: str, record: FlowStateRecord) -> str:
    return (
        f"{marker} record {record.id}  {record.method_name}  "
        f"{record.timestamp.isoformat()}"
    )


def diff_flow_states(
    db_path: str, flow_uuid: str, before_id: int, after_id: int, as_json: bool
) -> None:
    """Print what changed between two saved states of one flow.

    The database is only read: a missing file is an error, not something to
    create. Both record ids must belong to ``flow_uuid``; an id saved by a
    different flow is reported as missing.
    """
    from crewai.flow.persistence import compare_flow_states, read_flow_state

    if not os.path.isfile(db_path):
        raise click.ClickException(f"Flow state database not found: {db_path}")

    records: list[FlowStateRecord] = []
    for record_id in (before_id, after_id):
        record = read_flow_state(db_path, flow_uuid, record_id)
        if record is None:
            raise click.ClickException(
                f"Flow {flow_uuid} has no state record {record_id} in {db_path}"
            )
        records.append(record)
    before, after = records

    diff = compare_flow_states(before.state, after.state)

    if as_json:
        payload = {
            "flow_uuid": flow_uuid,
            "before": _record_summary(before),
            "after": _record_summary(after),
            **diff.to_dict(),
        }
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(_record_header("---", before))
    click.echo(_record_header("+++", after))
    click.echo()
    click.echo(diff.render_text())
