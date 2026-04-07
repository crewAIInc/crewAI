"""CLI commands for inspecting checkpoint files."""

from __future__ import annotations

from datetime import datetime
import glob
import json
import os
from typing import Any

import click


def _find_checkpoints(location: str) -> list[str]:
    """Find checkpoint files in a directory, sorted newest first."""
    pattern = os.path.join(location, "*.json")
    return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)


def _load_metadata(path: str) -> dict[str, Any]:
    """Load checkpoint metadata without full deserialization."""
    with open(path) as f:
        data = json.load(f)

    entities = data.get("entities", [])
    event_count = len(data.get("event_record", {}).get("nodes", {}))

    parsed_entities: list[dict[str, Any]] = []
    for entity in entities:
        tasks = entity.get("tasks", [])
        completed = sum(1 for t in tasks if t.get("output") is not None)
        info: dict[str, Any] = {
            "type": entity.get("entity_type", "unknown"),
            "name": entity.get("name"),
            "id": entity.get("id"),
        }
        if tasks:
            info["tasks_completed"] = completed
            info["tasks_total"] = len(tasks)
            info["tasks"] = [
                {
                    "description": t.get("description", ""),
                    "completed": t.get("output") is not None,
                }
                for t in tasks
            ]
        parsed_entities.append(info)

    return {
        "path": path,
        "size": os.path.getsize(path),
        "event_count": event_count,
        "entities": parsed_entities,
    }


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


def _ts_from_filename(path: str) -> str | None:
    """Extract timestamp from checkpoint filename."""
    name = os.path.basename(path).split("_")[0]
    try:
        dt = datetime.strptime(name, "%Y%m%dT%H%M%S")
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def list_checkpoints(location: str) -> None:
    """List all checkpoints in a directory."""
    files = _find_checkpoints(location)

    if not files:
        click.echo(f"No checkpoints found in {location}")
        return

    click.echo(f"Found {len(files)} checkpoint(s) in {location}\n")

    for path in files:
        ts = _ts_from_filename(path) or "unknown"
        size = _format_size(os.path.getsize(path))
        name = os.path.basename(path)

        try:
            meta = _load_metadata(path)
            parts = []
            for ent in meta["entities"]:
                etype = ent.get("type", "unknown")
                ename = ent.get("name", "")
                completed = ent.get("tasks_completed")
                total = ent.get("tasks_total")
                if completed is not None and total is not None:
                    parts.append(f"{etype}:{ename} [{completed}/{total} tasks]")
                else:
                    parts.append(f"{etype}:{ename}")
            summary = ", ".join(parts) if parts else "empty"
        except Exception:
            summary = "unreadable"

        click.echo(f"  {name}  {ts}  {size}  {summary}")


def info_checkpoint(path: str) -> None:
    """Show details of a single checkpoint."""
    if os.path.isdir(path):
        files = _find_checkpoints(path)
        if not files:
            click.echo(f"No checkpoints found in {path}")
            return
        path = files[0]
        click.echo(f"Latest checkpoint: {os.path.basename(path)}\n")

    if not os.path.isfile(path):
        click.echo(f"File not found: {path}")
        return

    try:
        meta = _load_metadata(path)
    except Exception as exc:
        click.echo(f"Failed to read checkpoint: {exc}")
        return

    ts = _ts_from_filename(path) or "unknown"
    click.echo(f"File:    {meta['path']}")
    click.echo(f"Time:    {ts}")
    click.echo(f"Size:    {_format_size(meta['size'])}")
    click.echo(f"Events:  {meta['event_count']}")

    for ent in meta["entities"]:
        eid = str(ent.get("id", ""))[:8]
        click.echo(f"\n  {ent['type']}: {ent.get('name', 'unnamed')} ({eid}...)")

        tasks = ent.get("tasks")
        if isinstance(tasks, list):
            click.echo(
                f"  Tasks: {ent['tasks_completed']}/{ent['tasks_total']} completed"
            )
            for i, task in enumerate(tasks):
                status = "done" if task.get("completed") else "pending"
                desc = str(task.get("description", ""))
                if len(desc) > 70:
                    desc = desc[:67] + "..."
                click.echo(f"    {i + 1}. [{status}] {desc}")
