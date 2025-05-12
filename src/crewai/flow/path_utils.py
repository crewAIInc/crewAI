"""Path utilities for secure file operations in CrewAI flow module.

This module provides utilities for secure path handling to prevent directory
traversal attacks and ensure paths remain within allowed boundaries.
"""

from pathlib import Path


def safe_path_join(*parts: str, root: str | Path | None = None) -> str:
    """Safely join path components and ensure the result is within allowed boundaries.

    Parameters
    ----------
    *parts : str
        Variable number of path components to join.
    root : Union[str, Path, None], optional
        Root directory to use as base. If None, uses current working directory.

    Returns
    -------
    str
        String representation of the resolved path.

    Raises
    ------
    ValueError
        If the resulting path would be outside the root directory
        or if any path component is invalid.

    """
    if not parts:
        msg = "No path components provided"
        raise ValueError(msg)

    try:
        # Convert all parts to strings and clean them
        clean_parts = [str(part).strip() for part in parts if part]
        if not clean_parts:
            msg = "No valid path components provided"
            raise ValueError(msg)

        # Establish root directory
        root_path = Path(root).resolve() if root else Path.cwd()

        # Join and resolve the full path
        full_path = Path(root_path, *clean_parts).resolve()

        # Check if the resolved path is within root
        if not str(full_path).startswith(str(root_path)):
            msg = f"Invalid path: Potential directory traversal. Path must be within {root_path}"
            raise ValueError(
                msg,
            )

        return str(full_path)

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        msg = f"Invalid path components: {e!s}"
        raise ValueError(msg)


def validate_path_exists(path: str | Path, file_type: str = "file") -> str:
    """Validate that a path exists and is of the expected type.

    Parameters
    ----------
    path : Union[str, Path]
        Path to validate.
    file_type : str, optional
        Expected type ('file' or 'directory'), by default 'file'.

    Returns
    -------
    str
        Validated path as string.

    Raises
    ------
    ValueError
        If path doesn't exist or is not of expected type.

    """
    try:
        path_obj = Path(path).resolve()

        if not path_obj.exists():
            msg = f"Path does not exist: {path}"
            raise ValueError(msg)

        if file_type == "file" and not path_obj.is_file():
            msg = f"Path is not a file: {path}"
            raise ValueError(msg)
        if file_type == "directory" and not path_obj.is_dir():
            msg = f"Path is not a directory: {path}"
            raise ValueError(msg)

        return str(path_obj)

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        msg = f"Invalid path: {e!s}"
        raise ValueError(msg)


def list_files(directory: str | Path, pattern: str = "*") -> list[str]:
    """Safely list files in a directory matching a pattern.

    Parameters
    ----------
    directory : Union[str, Path]
        Directory to search in.
    pattern : str, optional
        Glob pattern to match files against, by default "*".

    Returns
    -------
    List[str]
        List of matching file paths.

    Raises
    ------
    ValueError
        If directory is invalid or inaccessible.

    """
    try:
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            msg = f"Not a directory: {directory}"
            raise ValueError(msg)

        return [str(p) for p in dir_path.glob(pattern) if p.is_file()]

    except Exception as e:
        if isinstance(e, ValueError):
            raise
        msg = f"Error listing files: {e!s}"
        raise ValueError(msg)
