import logging
import os
import sys
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def add_project_to_path(project_dir: Optional[str] = None) -> None:
    """
    Add the project directory to the Python path to resolve module imports.
    
    This function is especially useful when starting flows from custom scripts
    outside of the CLI command context, to avoid ModuleNotFoundError.
    
    Args:
        project_dir: Optional path to the project directory. If not provided,
                     the current working directory is used.
    
    Raises:
        ValueError: If the provided directory does not exist.
    
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
        logger.debug(f"Using current working directory: {project_dir}")
    else:
        logger.debug(f"Using provided directory: {project_dir}")
    
    project_path = Path(project_dir).resolve()
    
    if not project_path.exists():
        raise ValueError(f"Invalid directory: {project_dir} does not exist")
    
    if (project_path / "src").exists() and (project_path / "src").is_dir():
        logger.debug(f"Found 'src' directory in {project_path}")
        if str(project_path) not in sys.path:
            sys.path.insert(0, str(project_path))
            logger.debug(f"Added {project_path} to sys.path")
        
        if str(project_path / "src") not in sys.path:
            sys.path.insert(0, str(project_path / "src"))
            logger.debug(f"Added {project_path / 'src'} to sys.path")
    else:
        logger.debug(f"No 'src' directory found in {project_path}")
        if str(project_path) not in sys.path:
            sys.path.insert(0, str(project_path))
            logger.debug(f"Added {project_path} to sys.path")
