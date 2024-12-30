"""Utilities for safe path handling in flow visualization.

This module provides a comprehensive set of utilities for secure path handling,
including path joining, validation, and normalization. It helps prevent common
security issues like directory traversal attacks while providing a consistent
interface for path operations.
"""

import os
from pathlib import Path
from typing import Union, List, Optional


def safe_path_join(base_dir: Union[str, Path], filename: str) -> str:
    """Safely join base directory with filename, preventing directory traversal.
    
    Args:
        base_dir: Base directory path
        filename: Filename or path to join with base_dir
        
    Returns:
        str: Safely joined absolute path
        
    Raises:
        ValueError: If resulting path would escape base_dir or contains dangerous patterns
        TypeError: If inputs are not strings or Path objects
        OSError: If path resolution fails
    """
    if not isinstance(base_dir, (str, Path)):
        raise TypeError("base_dir must be a string or Path object")
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
        
    # Check for dangerous patterns
    dangerous_patterns = ['..', '~', '*', '?', '|', '>', '<', '$', '&', '`']
    if any(pattern in filename for pattern in dangerous_patterns):
        raise ValueError(f"Invalid filename: Contains dangerous pattern")
        
    try:
        base_path = Path(base_dir).resolve(strict=True)
        full_path = Path(base_path, filename).resolve(strict=True)
        
        if not str(full_path).startswith(str(base_path)):
            raise ValueError(
                f"Invalid path: {filename} would escape base directory {base_dir}"
            )
            
        return str(full_path)
    except OSError as e:
        raise OSError(f"Failed to resolve path: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to process paths: {str(e)}")


def normalize_path(path: Union[str, Path]) -> str:
    """Normalize a path by resolving symlinks and removing redundant separators.
    
    Args:
        path: Path to normalize
        
    Returns:
        str: Normalized absolute path
        
    Raises:
        TypeError: If path is not a string or Path object
        OSError: If path resolution fails
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path object")
        
    try:
        return str(Path(path).resolve(strict=True))
    except OSError as e:
        raise OSError(f"Failed to normalize path: {str(e)}")


def validate_path_components(components: List[str]) -> None:
    """Validate path components for potentially dangerous patterns.
    
    Args:
        components: List of path components to validate
        
    Raises:
        TypeError: If components is not a list or contains non-string items
        ValueError: If any component contains dangerous patterns
    """
    if not isinstance(components, list):
        raise TypeError("components must be a list")
        
    dangerous_patterns = ['..', '~', '*', '?', '|', '>', '<', '$', '&', '`']
    for component in components:
        if not isinstance(component, str):
            raise TypeError(f"Path component '{component}' must be a string")
        if any(pattern in component for pattern in dangerous_patterns):
            raise ValueError(f"Invalid path component '{component}': Contains dangerous pattern")


def validate_file_path(path: Union[str, Path], must_exist: bool = True) -> str:
    """Validate a file path for security and existence.
    
    Args:
        path: File path to validate
        must_exist: Whether the file must exist (default: True)
        
    Returns:
        str: Validated absolute path
        
    Raises:
        ValueError: If path is invalid or file doesn't exist when required
        TypeError: If path is not a string or Path object
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path object")
        
    try:
        resolved_path = Path(path).resolve()
        
        if must_exist and not resolved_path.is_file():
            raise ValueError(f"File not found: {path}")
            
        return str(resolved_path)
    except Exception as e:
        raise ValueError(f"Invalid file path {path}: {str(e)}")
