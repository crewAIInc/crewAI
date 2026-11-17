"""`crewai eval` and the run it starts to evaluate.

When a project has no traced run to grade, or its code changed since the last
one, `crewai eval` runs the crew itself and evaluates that run. The run has to
know it was started for that: its trace is what the evaluation reads, so it is
traced and sent to CrewAI AMP without asking again, whoever is at the keyboard
(a person, or a coding agent with no terminal at all). Running `crewai eval` is
the request to have the run graded, and the grade needs the trace.

The run usually happens in a child process (`uv run …` in the project's own
environment), which only the environment reaches. The child loads the project's
`.env` over that environment, so a variable alone would be a switch any project
could flip: every run approved for upload, with nobody having asked. Its value
is therefore a random token, and it counts only while a file of that name
exists in this user's crewAI data directory. The command writes the file before
the run and removes it after, and a `.env` cannot create a file. Nor can it
move the directory: it is found from this account's home in the account
database, never from `HOME` or `XDG_DATA_HOME`, which a `.env` can set.

The file also carries what the command decided before the project's `.env` was
loaded, which the run must not take from that `.env`: the AMP origins the saved
login may be sent to, and whether this command just turned tracing on.

The CLI's run app (`crewai_cli.crew_run_tui`) keeps its own copy of the
variable's name, the folder and the file's shape rather than importing them,
because it also runs inside a project's environment whose crewai-core may be
older; a test pins the two to the same values.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import contextlib
from contextlib import contextmanager
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
import secrets
import sys
from typing import Any
from urllib.parse import urlparse

import appdirs


AWAITING_ENV = "CREWAI_EVAL_AWAITING_RUN"
_TOKEN = re.compile(r"[0-9a-f]{32}")


def _account_home() -> Path:
    """This account's home from the account database, never `$HOME`."""
    if os.name == "posix":
        with contextlib.suppress(ImportError, KeyError, OSError):
            import pwd

            return Path(pwd.getpwuid(os.getuid()).pw_dir)
    return Path.home()


def awaiting_dir() -> Path:
    """Where the tokens live: this user's crewAI data directory, wherever the
    environment says home is. Windows reads it from the known-folder API."""
    if os.name == "nt":
        return Path(appdirs.user_data_dir("crewai", "CrewAI")) / "eval-awaiting"
    home = _account_home()
    if sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = home / ".local" / "share"  # XDG_DATA_HOME ignored on purpose
    return base / "crewai" / "eval-awaiting"


def awaited_evaluation() -> dict[str, Any] | None:
    """What `crewai eval` said about this run, when it started it: None for a
    run nobody is evaluating. Never raises: a directory or a file that cannot
    be read means no evaluation is waiting; an older command's empty file
    carries nothing but the request itself."""
    value = os.environ.get(AWAITING_ENV) or ""
    if not _TOKEN.fullmatch(value):
        return None
    try:
        text = (awaiting_dir() / value).read_text(encoding="utf-8")
    except OSError:
        return None
    with contextlib.suppress(ValueError):
        said = json.loads(text)
        if isinstance(said, dict):
            return said
    return {}


def run_awaited_by_evaluation() -> bool:
    """Was this run started by `crewai eval`, to be evaluated?"""
    return awaited_evaluation() is not None


def evaluation_trusts_amp(base_url: str | None) -> bool:
    """May the saved login go to BASE_URL for this run? Only to an AMP origin
    `crewai eval` trusted before the project's `.env` was loaded, over HTTPS or
    plain HTTP to this machine — the rule the command applies to its own
    requests. A run nobody is evaluating is not this rule's to decide."""
    said = awaited_evaluation()
    if said is None:
        return True
    origin = amp_origin(base_url)
    trusted = said.get("trusted_amp_origins")
    return (
        origin is not None
        and isinstance(trusted, list)
        and origin in trusted
        and _encrypted(origin)
    )


def amp_origin(url: str | None) -> str | None:
    parsed = urlparse(str(url or ""))
    return (
        f"{parsed.scheme}://{parsed.netloc}".lower()
        if parsed.scheme and parsed.netloc
        else None
    )


def _encrypted(origin: str) -> bool:
    parsed = urlparse(origin)
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    hostname = (parsed.hostname or "").rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


@contextmanager
def awaiting_evaluation(
    *, trusted_amp_origins: Iterable[str] = (), tracing_turned_on: bool = False
) -> Iterator[bool]:
    """Mark the runs started inside this block as awaited by `crewai eval`.

    Yields whether the mark took: without its file a child run cannot be told,
    and then it is traced only as the project would trace any run. The previous
    value of the variable is put back after, and the file removed.
    """
    before = os.environ.get(AWAITING_ENV)
    marker: Path | None = None
    with contextlib.suppress(OSError):
        nonce = secrets.token_hex(16)
        awaiting_dir().mkdir(parents=True, exist_ok=True)
        path = awaiting_dir() / nonce
        with path.open("x", encoding="utf-8") as out:
            json.dump(
                {
                    "trusted_amp_origins": sorted(set(trusted_amp_origins)),
                    "tracing_turned_on": tracing_turned_on,
                },
                out,
            )
        marker = path
        os.environ[AWAITING_ENV] = nonce
    try:
        yield marker is not None
    finally:
        if marker is not None:
            with contextlib.suppress(OSError):
                marker.unlink()
        if before is None:
            os.environ.pop(AWAITING_ENV, None)
        else:
            os.environ[AWAITING_ENV] = before
