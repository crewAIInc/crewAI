import os
import sys
from pathlib import Path
from typing import Optional


def add_project_to_path(project_dir: Optional[str] = None) -> None:
    """
    Add the project directory to the Python path to resolve module imports.
    
    This function is especially useful when starting flows from custom scripts
    outside of the CLI command context, to avoid ModuleNotFoundError.
    
    Args:
        project_dir: Optional path to the project directory. If not provided,
                     the current working directory is used.
    
    Example:
        ```python
        from crewai.utilities.path_utils import add_project_to_path
        
        add_project_to_path()
        
        from your_project.main import YourFlow
        
        flow = YourFlow()
        flow.kickoff()
        ```
    """
    if project_dir is None:
        project_dir = os.getcwd()
    
    project_path = Path(project_dir).resolve()
    
    if (project_path / "src").exists() and (project_path / "src").is_dir():
        if str(project_path) not in sys.path:
            sys.path.insert(0, str(project_path))
        
        if str(project_path / "src") not in sys.path:
            sys.path.insert(0, str(project_path / "src"))
    else:
        if str(project_path) not in sys.path:
            sys.path.insert(0, str(project_path))
