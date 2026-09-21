"""The last traced run of a project, recorded for `crewai eval`.

Once a run's spans have reached Wharf, crewAI writes ``.crewai/last_run.json``
in the project directory: the execution id, when the run started and ended,
and whether it was traced anonymously or under an account. Nothing is
printed — the id is internal — and ``crewai eval`` reads it back to evaluate
the run without the user pasting anything.

Inside a deployment nothing is recorded, and no check here is what stops it:
the platform kicks off with ``tracing`` off, so crewAI never starts a trace
session of its own, never builds a ``GrantSpanExporter``, and never reaches
this module.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any


try:  # POSIX only; without it the write below simply is not serialised
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LAST_RUN_DIR = ".crewai"
LAST_RUN_FILE = "last_run.json"
LOCK_FILE = "last_run.lock"


def project_dir() -> Path:
    """The run's project: the working directory, as every project-local path in crewAI."""
    return Path.cwd()


def recording_enabled() -> bool:
    """Off under the test suite, so kickoff-level tests leave no file behind.

    Nothing else is checked. A deployment is kept out by not getting here at
    all, and the platform's integration token is NOT a deployment marker — it
    is a credential `crewai create crew` writes into a project's own `.env`
    for platform tools, so treating it as one would stop recording the runs of
    every developer who uses them."""
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
    or the write failed. Never raises — a run is never failed by this.

    "Last" means the run that FINISHED last: a record already there for a run
    with a later `finished_at` is kept, so two crews finishing together in one
    project leave the later-finished one whichever writer gets to the file
    last. With no completion time to compare, the write stands."""
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
            # Reading what is there, deciding, and replacing are one step: another
            # process must not slip a newer record in between, or this write would
            # compare against a record that is already gone and overwrite it.
            with _exclusive(path.parent):
                if _keep(record, read_last_run(path.parent.parent)):
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


@contextlib.contextmanager
def _exclusive(directory: Path) -> Iterator[None]:
    """One writer at a time in this project, across processes: a lock file beside
    the record, held over the read, the comparison and the replace.

    The locking never fails a write. No `flock` at all (Windows), a lock file
    that cannot be opened, or a filesystem that refuses the lock (some network
    mounts) each leave the write to go ahead unserialised — the record is a
    convenience pointer for `crewai eval`, and having it unserialised beats not
    having it."""
    if fcntl is None:
        yield
        return
    try:
        handle = open(directory / LOCK_FILE, "a")
    except OSError:
        yield
        return
    locked = False
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            locked = True
        except OSError as error:
            logger.debug("Could not lock %s: %s", directory / LOCK_FILE, error)
        yield
    finally:
        if locked:
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _keep(record: dict[str, Any], existing: dict[str, Any] | None) -> bool:
    """Does RECORD replace EXISTING? Only a record we can PROVE finished later
    holds the file: a different run with a `finished_at` after ours. Everything
    else — no record there, the same run recorded again, a missing completion
    time on either side, a tie to the millisecond — leaves the write to stand,
    so the last run recorded is the one a reader gets."""
    if not existing or existing.get("execution_id") == record["execution_id"]:
        return True
    ours, theirs = record["finished_at"], existing.get("finished_at")
    if not isinstance(ours, str) or not isinstance(theirs, str):
        return True  # nothing to order by: the write stands
    # Both ISO 8601 in UTC at the same precision, so text order is time order;
    # a tie means two runs finished in the same millisecond and either is a fair "last run".
    return ours >= theirs


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
