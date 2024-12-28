import os
from pathlib import Path

import appdirs


def db_storage_path():
    app_name = get_project_directory_name()
    app_author = "CrewAI"

    data_dir = Path(appdirs.user_data_dir(app_name, app_author))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_project_directory_name():
    project_directory_name = os.environ.get("CREWAI_STORAGE_DIR")

    if project_directory_name:
        return project_directory_name
    else:
        cwd = Path.cwd()
        project_directory_name = cwd.name
        return project_directory_name

def get_default_storage_path(storage_type: str) -> Path:
    """Returns the default storage path for a given storage type.
    
    Args:
        storage_type: Type of storage ('ltm', 'kickoff', 'rag')
        
    Returns:
        Path: Default storage path for the specified type
        
    Raises:
        ValueError: If storage_type is not recognized
    """
    base_path = db_storage_path()
    
    if storage_type == 'ltm':
        return base_path / 'latest_long_term_memories.db'
    elif storage_type == 'kickoff':
        return base_path / 'latest_kickoff_task_outputs.db'
    elif storage_type == 'rag':
        return base_path
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
