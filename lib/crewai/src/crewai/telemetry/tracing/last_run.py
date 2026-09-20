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
    """Off under the test suite, so kickoff-level tests leave no file behind."""
    return os.environ.get("CREWAI_TESTING", "").lower() != "true"


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
    or the write failed. Never raises — a run is never failed by this."""
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
            os.replace(temporary, path)
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
