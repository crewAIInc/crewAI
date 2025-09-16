import os
from pathlib import Path

import appdirs

"""Path management utilities for CrewAI storage and configuration."""

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


def get_project_directory_name():
    """Returns the current project directory name."""
    project_directory_name = os.environ.get("CREWAI_STORAGE_DIR")

    if project_directory_name:
        return project_directory_name
    else:
        cwd = Path.cwd()
        project_directory_name = cwd.name
        return project_directory_name


def get_knowledge_directory():
    """Returns the knowledge directory path from environment variable or default."""
    knowledge_dir = os.environ.get("CREWAI_KNOWLEDGE_FILE_DIR")
    
    if knowledge_dir:
        knowledge_path = Path(knowledge_dir)
        if not knowledge_path.exists():
            raise ValueError(f"Knowledge directory does not exist: {knowledge_dir}")
        return str(knowledge_path)
    else:
        return "knowledge"
