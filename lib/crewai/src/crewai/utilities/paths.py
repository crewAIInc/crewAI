"""Path management utilities for CrewAI storage and configuration."""

import os
from pathlib import Path

import appdirs


def db_storage_path() -> str:
    """Returns the path for SQLite database storage.

    Returns:
        str: Full path to the SQLite database file
    """
    app_name = get_project_directory_name()
    app_author = "CrewAI"

    data_dir = Path(appdirs.user_data_dir(app_name, app_author))
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def get_project_directory_name() -> str:
    """Returns the current project directory name."""
    return os.environ.get("CREWAI_STORAGE_DIR", Path.cwd().name)
