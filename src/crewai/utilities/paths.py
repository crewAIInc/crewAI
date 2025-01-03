import os
from pathlib import Path

import appdirs

"""Path management utilities for CrewAI storage and configuration."""

def db_storage_path():
    """Returns the path for database storage."""
    app_name = get_project_directory_name()
    app_author = "CrewAI"

    data_dir = Path(appdirs.user_data_dir(app_name, app_author))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_project_directory_name():
    """Returns the current project directory name."""
    project_directory_name = os.environ.get("CREWAI_STORAGE_DIR")

    if project_directory_name:
        return project_directory_name
    else:
        cwd = Path.cwd()
        project_directory_name = cwd.name
        return project_directory_name
