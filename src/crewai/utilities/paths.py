"""Path management utilities for CrewAI storage and configuration."""

import os
from pathlib import Path


def db_storage_path() -> str:
    """Returns the path for SQLite database storage.

    Returns:
        str: Full path to the SQLite database file
    """
    storage_dir = os.environ.get("CREWAI_STORAGE_DIR")

    if storage_dir:
        data_dir = Path(storage_dir)
    else:
        data_dir = Path.cwd() / "db"

    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir)


def get_project_directory_name() -> str:
    """Returns the current project directory name."""
    project_directory_name = os.environ.get("CREWAI_STORAGE_DIR")

    if project_directory_name:
        return project_directory_name
    cwd = Path.cwd()
    return cwd.name
