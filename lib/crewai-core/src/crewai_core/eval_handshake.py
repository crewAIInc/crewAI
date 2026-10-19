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
the run and removes it after, and a `.env` cannot create a file.

The CLI's run app (`crewai_cli.crew_run_tui`) keeps its own copy of the
variable's name and the folder rather than importing them, because it also runs
inside a project's environment whose crewai-core may be older; a test pins the
two to the same values.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
from contextlib import contextmanager
import os
from pathlib import Path
import re
import secrets

import appdirs


AWAITING_ENV = "CREWAI_EVAL_AWAITING_RUN"
_TOKEN = re.compile(r"[0-9a-f]{32}")


def awaiting_dir() -> Path:
    """Where the tokens live: this user's crewAI data directory."""
    return Path(appdirs.user_data_dir("crewai", "CrewAI")) / "eval-awaiting"


def run_awaited_by_evaluation() -> bool:
    """Was this run started by `crewai eval`, to be evaluated?

    True only while the environment carries a token AND the command that set it
    still holds its file. Never raises: a directory that cannot be read means
    no evaluation is waiting.
    """
    value = os.environ.get(AWAITING_ENV) or ""
    if not _TOKEN.fullmatch(value):
        return False
    try:
        return (awaiting_dir() / value).is_file()
    except OSError:
        return False


@contextmanager
def awaiting_evaluation() -> Iterator[bool]:
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
        (awaiting_dir() / nonce).touch(exist_ok=False)
        marker = awaiting_dir() / nonce
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
