"""The last traced run of a project, recorded for `crewai eval`.

Once a run's spans have reached Wharf, crewAI writes ``.crewai/last_run.json``
in the project directory: the execution id, when the run started and ended,
and whether it was traced anonymously or under an account. Nothing is
printed — the id is internal — and ``crewai eval`` reads it back to evaluate
the run without the user pasting anything.

Inside a deployment nothing is recorded: the platform binds the execution
before crewAI would start its own tracing, so this code never runs there.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any


logger = logging.getLogger(__name__)

LAST_RUN_DIR = ".crewai"
LAST_RUN_FILE = "last_run.json"


def project_dir() -> Path:
    """The run's project: the working directory, as every project-local path in crewAI."""
    return Path.cwd()


def recording_enabled() -> bool:
    """Off under the test suite, so kickoff-level tests leave no file behind,
    and off inside a deployment — the platform's integration token marks one —
    so a run there never writes a file the platform never reads. (A deployment
    normally never gets here at all: the host binds the trace before crewAI
    would start its own; this is the guard for a container that did not.)"""
    from crewai.context import get_platform_integration_token

    if os.environ.get("CREWAI_TESTING", "").lower() == "true":
        return False
    return get_platform_integration_token() is None


def last_run_path(directory: Path | None = None) -> Path:
    return (directory or project_dir()) / LAST_RUN_DIR / LAST_RUN_FILE


def _iso(nanoseconds: int | None) -> str | None:
    if nanoseconds is None:
        return None
    return datetime.fromtimestamp(nanoseconds / 1e9, tz=timezone.utc).isoformat(
        timespec="milliseconds"
    )


def record_last_run(
    *,
    execution_id: str,
    tier: str | None,
    started_at_ns: int | None,
    finished_at_ns: int | None,
    amp_base_url: str | None,
) -> Path | None:
    """Write the record atomically; the path, or None when recording is off
    or the write failed. Never raises — a run is never failed by this.

    "Last" means the run that FINISHED last: a record already there for a run
    that finished later — or in the same millisecond — is kept, so two crews
    finishing together in one project leave the newer one whichever writer
    gets to the file last."""
    if not recording_enabled():
        return None
    record: dict[str, Any] = {
        "execution_id": execution_id,
        "tier": tier,
        "started_at": _iso(started_at_ns),
        "finished_at": _iso(finished_at_ns),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "amp_base_url": amp_base_url,
    }
    path: Path | None = None
    try:
        path = last_run_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # A temporary file of its own per writer: two crews finishing together in one
        # project must not write through the same name, or one record is lost.
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record, indent=2) + "\n")
            if _newer_than(record, read_last_run(path.parent.parent)):
                os.replace(temporary, path)
            else:
                logger.debug(
                    "A run that finished later is already recorded in %s", path
                )
                os.unlink(temporary)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
    except OSError as error:  # a vanished cwd fails last_run_path() too; the run is never failed by this
        logger.debug(
            "Could not record the last run in %s: %s",
            path or ".crewai/last_run.json",
            error,
        )
        return None
    return path


def _newer_than(record: dict[str, Any], existing: dict[str, Any] | None) -> bool:
    """Is RECORD the later-finished run? A missing or unreadable existing record,
    or one without a comparable time, never wins over the run just finished; a
    tie (the same millisecond) keeps what is there — the times are stored to
    the millisecond, and no finer order is worth a field in the file."""
    if not existing or existing.get("execution_id") == record["execution_id"]:
        return True
    ours = record["finished_at"] or record["recorded_at"]
    theirs = existing.get("finished_at") or existing.get("recorded_at")
    if not isinstance(theirs, str) or not isinstance(ours, str):
        return True
    # Both ISO 8601 in UTC at the same precision: text order is time order.
    return ours > theirs


def read_last_run(directory: Path | None = None) -> dict[str, Any] | None:
    """The record, or None when there is none or it cannot be read."""
    path = last_run_path(directory)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict) or not loaded.get("execution_id"):
        return None
    return loaded
