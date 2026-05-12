"""Path management utilities for CrewAI storage and configuration."""

from __future__ import annotations

import os
from pathlib import Path

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
