"""CLI commands for inspecting checkpoint files."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import glob
import json
import os
import re
import sqlite3
from typing import Any

import click


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_\-]*)}")


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

_DELETE_OLDER_THAN = """
DELETE FROM checkpoints
WHERE created_at < ?
"""

_DELETE_KEEP_N = """
DELETE FROM checkpoints WHERE rowid NOT IN (
    SELECT rowid FROM checkpoints ORDER BY rowid DESC LIMIT ?
)
"""

_COUNT_CHECKPOINTS = "SELECT COUNT(*) FROM checkpoints"

_SELECT_LIKE = """
SELECT id, created_at, json(data)
FROM checkpoints
WHERE id LIKE ?
ORDER BY rowid DESC
"""


_DEFAULT_DIR = "./.checkpoints"
_DEFAULT_DB = "./.checkpoints.db"


def _detect_location(location: str) -> str:
    """Resolve the default checkpoint location.

    When the caller passes the default directory path, check whether a
    SQLite database exists at the conventional ``.db`` path and prefer it.
    """
    if (
        location == _DEFAULT_DIR
        and not os.path.exists(_DEFAULT_DIR)
        and os.path.exists(_DEFAULT_DB)
    ):
        return _DEFAULT_DB
    return location


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

    trigger_event = data.get("trigger")

    parsed_entities: list[dict[str, Any]] = []
    for entity in entities:
        tasks = entity.get("tasks", [])
        completed = sum(1 for t in tasks if t.get("output") is not None)
        info: dict[str, Any] = {
            "type": entity.get("entity_type", "unknown"),
            "name": entity.get("name"),
            "id": entity.get("id"),
        }

        raw_agents = entity.get("agents", [])
        agents_by_id: dict[str, dict[str, Any]] = {}
        parsed_agents: list[dict[str, Any]] = []
        for ag in raw_agents:
            agent_info: dict[str, Any] = {
                "id": ag.get("id", ""),
                "role": ag.get("role", ""),
                "goal": ag.get("goal", ""),
            }
            parsed_agents.append(agent_info)
            if ag.get("id"):
                agents_by_id[str(ag["id"])] = agent_info
        if parsed_agents:
            info["agents"] = parsed_agents

        if tasks:
            info["tasks_completed"] = completed
            info["tasks_total"] = len(tasks)
            parsed_tasks: list[dict[str, Any]] = []
            for t in tasks:
                task_info: dict[str, Any] = {
                    "description": t.get("description", ""),
                    "completed": t.get("output") is not None,
                    "output": (t.get("output") or {}).get("raw", ""),
                }
                task_agent = t.get("agent")
                if isinstance(task_agent, dict):
                    task_info["agent_role"] = task_agent.get("role", "")
                    task_info["agent_id"] = task_agent.get("id", "")
                elif isinstance(task_agent, str) and task_agent in agents_by_id:
                    task_info["agent_role"] = agents_by_id[task_agent].get("role", "")
                    task_info["agent_id"] = task_agent
                parsed_tasks.append(task_info)
            info["tasks"] = parsed_tasks

        if entity.get("entity_type") == "flow":
            completed_methods = entity.get("checkpoint_completed_methods")
            if completed_methods:
                info["completed_methods"] = sorted(completed_methods)
            state = entity.get("checkpoint_state")
            if isinstance(state, dict):
                info["flow_state"] = state

        parsed_entities.append(info)

    inputs: dict[str, Any] = {}
    for entity in entities:
        cp_inputs = entity.get("checkpoint_inputs")
        if isinstance(cp_inputs, dict) and cp_inputs:
            inputs = dict(cp_inputs)
            break

    for entity in entities:
        for task in entity.get("tasks", []):
            for field in (
                "checkpoint_original_description",
                "checkpoint_original_expected_output",
            ):
                text = task.get(field) or ""
                for match in _PLACEHOLDER_RE.findall(text):
                    if match not in inputs:
                        inputs[match] = ""
        for agent in entity.get("agents", []):
            for field in ("role", "goal", "backstory"):
                text = agent.get(field) or ""
                for match in _PLACEHOLDER_RE.findall(text):
                    if match not in inputs:
                        inputs[match] = ""

    branch = data.get("branch", "main")
    parent_id = data.get("parent_id")

    return {
        "source": source,
        "event_count": event_count,
        "trigger": trigger_event,
        "entities": parsed_entities,
        "branch": branch,
        "parent_id": parent_id,
        "inputs": inputs,
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
    pattern = os.path.join(location, "**", "*.json")
    results = []
    for path in sorted(
        glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True
    ):
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
    pattern = os.path.join(location, "**", "*.json")
    files = sorted(
        glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True
    )
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
            meta["db"] = db_path
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
            row = conn.execute(_SELECT_LIKE, (f"%{checkpoint_id}%",)).fetchone()
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
    click.echo(f"Branch:  {meta.get('branch', 'main')}")
    parent_id = meta.get("parent_id")
    if parent_id:
        click.echo(f"Parent:  {parent_id}")

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


def _resolve_checkpoint(
    location: str, checkpoint_id: str | None
) -> dict[str, Any] | None:
    if _is_sqlite(location):
        if checkpoint_id:
            return _info_sqlite_id(location, checkpoint_id)
        return _info_sqlite_latest(location)
    if os.path.isdir(location):
        if checkpoint_id:
            from crewai.state.provider.json_provider import JsonProvider

            _json_provider: JsonProvider = JsonProvider()
            pattern: str = os.path.join(location, "**", "*.json")
            all_files: list[str] = glob.glob(pattern, recursive=True)
            matches: list[str] = [
                f for f in all_files if checkpoint_id in _json_provider.extract_id(f)
            ]
            matches.sort(key=os.path.getmtime, reverse=True)
            if matches:
                return _info_json_file(matches[0])
            return None
        return _info_json_latest(location)
    if os.path.isfile(location):
        return _info_json_file(location)
    return None


def _entity_type_from_meta(meta: dict[str, Any]) -> str:
    for ent in meta.get("entities", []):
        if ent.get("type") == "flow":
            return "flow"
        if ent.get("type") == "agent":
            return "agent"
    return "crew"


def resume_checkpoint(location: str, checkpoint_id: str | None) -> None:
    import asyncio

    meta: dict[str, Any] | None = _resolve_checkpoint(location, checkpoint_id)
    if meta is None:
        if checkpoint_id:
            click.echo(f"Checkpoint not found: {checkpoint_id}")
        else:
            click.echo(f"No checkpoints found in {location}")
        return

    restore_path: str = meta.get("path") or meta.get("source", "")
    if meta.get("db"):
        restore_path = f"{meta['db']}#{meta['name']}"

    click.echo(f"Resuming from: {meta.get('name', restore_path)}")
    _print_info(meta)
    click.echo()

    from crewai.state.checkpoint_config import CheckpointConfig

    config: CheckpointConfig = CheckpointConfig(restore_from=restore_path)
    entity_type: str = _entity_type_from_meta(meta)
    inputs: dict[str, Any] | None = meta.get("inputs") or None

    if entity_type == "flow":
        from crewai.flow.flow import Flow

        flow = Flow.from_checkpoint(config)
        result = asyncio.run(flow.kickoff_async(inputs=inputs))
    elif entity_type == "agent":
        from crewai.agent import Agent

        agent = Agent.from_checkpoint(config)
        result = asyncio.run(agent.akickoff(messages="Resume execution."))
    else:
        from crewai.crew import Crew

        crew = Crew.from_checkpoint(config)
        result = asyncio.run(crew.akickoff(inputs=inputs))

    click.echo(f"\nResult: {getattr(result, 'raw', result)}")


def _task_list_from_meta(meta: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for ent in meta.get("entities", []):
        tasks.extend(
            {
                "entity": ent.get("name", "unnamed"),
                "description": t.get("description", ""),
                "completed": t.get("completed", False),
                "output": t.get("output", ""),
            }
            for t in ent.get("tasks", [])
        )
    return tasks


def diff_checkpoints(location: str, id1: str, id2: str) -> None:
    meta1: dict[str, Any] | None = _resolve_checkpoint(location, id1)
    meta2: dict[str, Any] | None = _resolve_checkpoint(location, id2)

    if meta1 is None:
        click.echo(f"Checkpoint not found: {id1}")
        return
    if meta2 is None:
        click.echo(f"Checkpoint not found: {id2}")
        return

    name1: str = meta1.get("name", id1)
    name2: str = meta2.get("name", id2)

    click.echo(f"--- {name1}")
    click.echo(f"+++ {name2}")
    click.echo()

    fields: list[tuple[str, str]] = [
        ("Time", "ts"),
        ("Branch", "branch"),
        ("Trigger", "trigger"),
        ("Events", "event_count"),
    ]
    for label, key in fields:
        v1: str = str(meta1.get(key, ""))
        v2: str = str(meta2.get(key, ""))
        if v1 != v2:
            click.echo(f"  {label}:")
            click.echo(f"    - {v1}")
            click.echo(f"    + {v2}")

    inputs1: dict[str, Any] = meta1.get("inputs", {})
    inputs2: dict[str, Any] = meta2.get("inputs", {})
    all_keys: list[str] = sorted(set(list(inputs1.keys()) + list(inputs2.keys())))
    changed_inputs: list[tuple[str, Any, Any]] = [
        (k, inputs1.get(k, ""), inputs2.get(k, ""))
        for k in all_keys
        if inputs1.get(k) != inputs2.get(k)
    ]
    if changed_inputs:
        click.echo("\n  Inputs:")
        for key, v1, v2 in changed_inputs:
            click.echo(f"    {key}:")
            click.echo(f"      - {v1}")
            click.echo(f"      + {v2}")

    tasks1: list[dict[str, Any]] = _task_list_from_meta(meta1)
    tasks2: list[dict[str, Any]] = _task_list_from_meta(meta2)

    max_tasks: int = max(len(tasks1), len(tasks2))
    if max_tasks == 0:
        return

    click.echo("\n  Tasks:")
    for i in range(max_tasks):
        t1: dict[str, Any] | None = tasks1[i] if i < len(tasks1) else None
        t2: dict[str, Any] | None = tasks2[i] if i < len(tasks2) else None

        if t1 is None:
            desc: str = t2["description"][:60] if t2 else ""
            click.echo(f"    + {i + 1}. [new] {desc}")
            continue
        if t2 is None:
            desc = t1["description"][:60]
            click.echo(f"    - {i + 1}. [removed] {desc}")
            continue

        desc = str(t1["description"][:60])
        s1: str = "done" if t1["completed"] else "pending"
        s2: str = "done" if t2["completed"] else "pending"

        if s1 != s2:
            click.echo(f"    {i + 1}. {desc}")
            click.echo(f"      status: {s1} -> {s2}")

        out1: str = (t1.get("output") or "").strip()
        out2: str = (t2.get("output") or "").strip()
        if out1 != out2:
            if s1 == s2:
                click.echo(f"    {i + 1}. {desc}")
            preview1: str = (
                out1[:80] + ("..." if len(out1) > 80 else "") if out1 else "(empty)"
            )
            preview2: str = (
                out2[:80] + ("..." if len(out2) > 80 else "") if out2 else "(empty)"
            )
            click.echo("      output:")
            click.echo(f"        - {preview1}")
            click.echo(f"        + {preview2}")


def _parse_duration(value: str) -> timedelta:
    match: re.Match[str] | None = re.match(r"^(\d+)([dhm])$", value.strip())
    if not match:
        raise click.BadParameter(
            f"Invalid duration: {value!r}. Use format like '7d', '24h', or '30m'."
        )
    amount: int = int(match.group(1))
    unit: str = match.group(2)
    if unit == "d":
        return timedelta(days=amount)
    if unit == "h":
        return timedelta(hours=amount)
    return timedelta(minutes=amount)


def _prune_json(location: str, keep: int | None, older_than: timedelta | None) -> int:
    pattern: str = os.path.join(location, "**", "*.json")
    files: list[str] = sorted(
        glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True
    )
    if not files:
        return 0

    to_delete: set[str] = set()

    if keep is not None and len(files) > keep:
        to_delete.update(files[keep:])

    if older_than is not None:
        cutoff: datetime = datetime.now(timezone.utc) - older_than
        for path in files:
            mtime: datetime = datetime.fromtimestamp(
                os.path.getmtime(path), tz=timezone.utc
            )
            if mtime < cutoff:
                to_delete.add(path)

    deleted: int = 0
    for path in to_delete:
        try:
            os.remove(path)
            deleted += 1
        except OSError:  # noqa: PERF203
            pass

    for dirpath, dirnames, filenames in os.walk(location, topdown=False):
        if dirpath != location and not filenames and not dirnames:
            try:
                os.rmdir(dirpath)
            except OSError:
                pass

    return deleted


def _prune_sqlite(db_path: str, keep: int | None, older_than: timedelta | None) -> int:
    deleted: int = 0
    with sqlite3.connect(db_path) as conn:
        if older_than is not None:
            cutoff: str = (datetime.now(timezone.utc) - older_than).strftime(
                "%Y%m%dT%H%M%S"
            )
            cursor: sqlite3.Cursor = conn.execute(_DELETE_OLDER_THAN, (cutoff,))
            deleted += cursor.rowcount

        if keep is not None:
            cursor = conn.execute(_DELETE_KEEP_N, (keep,))
            deleted += cursor.rowcount

        conn.commit()
    return deleted


def prune_checkpoints(
    location: str, keep: int | None, older_than: str | None, dry_run: bool = False
) -> None:
    if keep is None and older_than is None:
        click.echo("Specify --keep N and/or --older-than DURATION (e.g. 7d, 24h)")
        return

    duration: timedelta | None = _parse_duration(older_than) if older_than else None

    deleted: int
    if _is_sqlite(location):
        if dry_run:
            with sqlite3.connect(location) as conn:
                total: int = conn.execute(_COUNT_CHECKPOINTS).fetchone()[0]
            click.echo(f"Would prune from {total} checkpoint(s) in {location}")
            return
        deleted = _prune_sqlite(location, keep, duration)
    elif os.path.isdir(location):
        if dry_run:
            files: list[str] = glob.glob(
                os.path.join(location, "**", "*.json"), recursive=True
            )
            click.echo(f"Would prune from {len(files)} checkpoint(s) in {location}")
            return
        deleted = _prune_json(location, keep, duration)
    else:
        click.echo(f"Not a directory or SQLite database: {location}")
        return
    click.echo(f"Pruned {deleted} checkpoint(s) from {location}")
