"""CLI commands for inspecting checkpoint files."""

from __future__ import annotations

from datetime import datetime
import glob
import json
import os
import sqlite3
from typing import Any

import click


_SQLITE_MAGIC = b"SQLite format 3\x00"

_SELECT_ALL = """
SELECT id, created_at, json(data)
FROM checkpoints
ORDER BY rowid DESC
"""

_SELECT_ONE = """
SELECT id, created_at, json(data)
FROM checkpoints
WHERE id = ?
"""

_SELECT_LATEST = """
SELECT id, created_at, json(data)
FROM checkpoints
ORDER BY rowid DESC
LIMIT 1
"""


def _is_sqlite(path: str) -> bool:
    """Check if a file is a SQLite database by reading its magic bytes."""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            return f.read(16) == _SQLITE_MAGIC
    except OSError:
        return False


def _parse_checkpoint_json(raw: str, source: str) -> dict[str, Any]:
    """Parse checkpoint JSON into metadata dict."""
    data = json.loads(raw)
    entities = data.get("entities", [])
    nodes = data.get("event_record", {}).get("nodes", {})
    event_count = len(nodes)

    trigger_event = None
    if nodes:
        last_node = max(
            nodes.values(),
            key=lambda n: n.get("event", {}).get("emission_sequence") or 0,
        )
        trigger_event = last_node.get("event", {}).get("type")

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
        "source": source,
        "event_count": event_count,
        "trigger": trigger_event,
        "entities": parsed_entities,
    }


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / 1024 / 1024:.1f}MB"


def _ts_from_name(name: str) -> str | None:
    """Extract timestamp from checkpoint ID or filename."""
    stem = os.path.basename(name).split("_")[0].removesuffix(".json")
    try:
        dt = datetime.strptime(stem, "%Y%m%dT%H%M%S")
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _entity_summary(entities: list[dict[str, Any]]) -> str:
    parts = []
    for ent in entities:
        etype = ent.get("type", "unknown")
        ename = ent.get("name", "")
        completed = ent.get("tasks_completed")
        total = ent.get("tasks_total")
        if completed is not None and total is not None:
            parts.append(f"{etype}:{ename} [{completed}/{total} tasks]")
        else:
            parts.append(f"{etype}:{ename}")
    return ", ".join(parts) if parts else "empty"


# --- JSON directory ---


def _list_json(location: str) -> list[dict[str, Any]]:
    pattern = os.path.join(location, "*.json")
    results = []
    for path in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True):
        name = os.path.basename(path)
        try:
            with open(path) as f:
                raw = f.read()
            meta = _parse_checkpoint_json(raw, source=name)
            meta["name"] = name
            meta["ts"] = _ts_from_name(name)
            meta["size"] = os.path.getsize(path)
            meta["path"] = path
        except Exception:
            meta = {"name": name, "ts": None, "size": 0, "entities": [], "source": name}
        results.append(meta)
    return results


def _info_json_latest(location: str) -> dict[str, Any] | None:
    pattern = os.path.join(location, "*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None
    path = files[0]
    with open(path) as f:
        raw = f.read()
    meta = _parse_checkpoint_json(raw, source=os.path.basename(path))
    meta["name"] = os.path.basename(path)
    meta["ts"] = _ts_from_name(path)
    meta["size"] = os.path.getsize(path)
    meta["path"] = path
    return meta


def _info_json_file(path: str) -> dict[str, Any]:
    with open(path) as f:
        raw = f.read()
    meta = _parse_checkpoint_json(raw, source=os.path.basename(path))
    meta["name"] = os.path.basename(path)
    meta["ts"] = _ts_from_name(path)
    meta["size"] = os.path.getsize(path)
    meta["path"] = path
    return meta


# --- SQLite ---


def _list_sqlite(db_path: str) -> list[dict[str, Any]]:
    results = []
    with sqlite3.connect(db_path) as conn:
        for row in conn.execute(_SELECT_ALL):
            checkpoint_id, created_at, raw = row
            try:
                meta = _parse_checkpoint_json(raw, source=checkpoint_id)
                meta["name"] = checkpoint_id
                meta["ts"] = _ts_from_name(checkpoint_id) or created_at
            except Exception:
                meta = {
                    "name": checkpoint_id,
                    "ts": created_at,
                    "entities": [],
                    "source": checkpoint_id,
                }
            results.append(meta)
    return results


def _info_sqlite_latest(db_path: str) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(_SELECT_LATEST).fetchone()
    if not row:
        return None
    checkpoint_id, created_at, raw = row
    meta = _parse_checkpoint_json(raw, source=checkpoint_id)
    meta["name"] = checkpoint_id
    meta["ts"] = _ts_from_name(checkpoint_id) or created_at
    meta["db"] = db_path
    return meta


def _info_sqlite_id(db_path: str, checkpoint_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(_SELECT_ONE, (checkpoint_id,)).fetchone()
    if not row:
        return None
    cid, created_at, raw = row
    meta = _parse_checkpoint_json(raw, source=cid)
    meta["name"] = cid
    meta["ts"] = _ts_from_name(cid) or created_at
    meta["db"] = db_path
    return meta


# --- Public API ---


def list_checkpoints(location: str) -> None:
    """List all checkpoints at a location."""
    if _is_sqlite(location):
        entries = _list_sqlite(location)
        label = f"SQLite: {location}"
    elif os.path.isdir(location):
        entries = _list_json(location)
        label = location
    else:
        click.echo(f"Not a directory or SQLite database: {location}")
        return

    if not entries:
        click.echo(f"No checkpoints found in {label}")
        return

    click.echo(f"Found {len(entries)} checkpoint(s) in {label}\n")

    for entry in entries:
        ts = entry.get("ts") or "unknown"
        name = entry.get("name", "")
        size = _format_size(entry["size"]) if "size" in entry else ""
        trigger = entry.get("trigger") or ""
        summary = _entity_summary(entry.get("entities", []))
        parts = [name, ts]
        if size:
            parts.append(size)
        if trigger:
            parts.append(trigger)
        parts.append(summary)
        click.echo(f"  {'  '.join(parts)}")


def info_checkpoint(path: str) -> None:
    """Show details of a single checkpoint."""
    meta: dict[str, Any] | None = None

    # db_path#checkpoint_id format
    if "#" in path:
        db_path, checkpoint_id = path.rsplit("#", 1)
        if _is_sqlite(db_path):
            meta = _info_sqlite_id(db_path, checkpoint_id)
            if not meta:
                click.echo(f"Checkpoint not found: {checkpoint_id}")
                return

    # SQLite file — show latest
    if meta is None and _is_sqlite(path):
        meta = _info_sqlite_latest(path)
        if not meta:
            click.echo(f"No checkpoints in database: {path}")
            return
        click.echo(f"Latest checkpoint: {meta['name']}\n")

    # Directory — show latest JSON
    if meta is None and os.path.isdir(path):
        meta = _info_json_latest(path)
        if not meta:
            click.echo(f"No checkpoints found in {path}")
            return
        click.echo(f"Latest checkpoint: {meta['name']}\n")

    # Specific JSON file
    if meta is None and os.path.isfile(path):
        try:
            meta = _info_json_file(path)
        except Exception as exc:
            click.echo(f"Failed to read checkpoint: {exc}")
            return

    if meta is None:
        click.echo(f"Not found: {path}")
        return

    _print_info(meta)


def _print_info(meta: dict[str, Any]) -> None:
    ts = meta.get("ts") or "unknown"
    source = meta.get("path") or meta.get("db") or meta.get("source", "")
    click.echo(f"Source:  {source}")
    click.echo(f"Name:    {meta.get('name', '')}")
    click.echo(f"Time:    {ts}")
    if "size" in meta:
        click.echo(f"Size:    {_format_size(meta['size'])}")
    click.echo(f"Events:  {meta.get('event_count', 0)}")
    trigger = meta.get("trigger")
    if trigger:
        click.echo(f"Trigger: {trigger}")

    for ent in meta.get("entities", []):
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
