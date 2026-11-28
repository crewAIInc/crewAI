"""Path management utilities for CrewAI storage and configuration."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sys

import appdirs


def get_project_directory_name() -> str:
    """Return the current project directory name (or ``CREWAI_STORAGE_DIR``)."""
    return os.environ.get("CREWAI_STORAGE_DIR", Path.cwd().name)


def db_storage_path() -> str:
    """Return the path for SQLite database / app-data storage.

    Creates the directory if it does not exist.
    """
    app_name = get_project_directory_name()
    app_author = "CrewAI"

    data_dir = Path(appdirs.user_data_dir(app_name, app_author))
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def account_data_dir() -> Path:
    """The folder ``db_storage_path`` keeps each project's folder in, found for
    this account: from its home in the account database (on Windows, the
    known-folder API), never from ``HOME`` or ``XDG_DATA_HOME``, which a
    project's ``.env`` can set.
    """
    if os.name == "nt":
        return Path(appdirs.user_data_dir()) / "CrewAI"
    home = Path.home()
    if os.name == "posix":
        with contextlib.suppress(ImportError, KeyError, OSError):
            import pwd

            home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    if sys.platform == "darwin":
        return home / "Library" / "Application Support"
    return home / ".local" / "share"  # XDG_DATA_HOME ignored on purpose
