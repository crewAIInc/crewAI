"""Path validation to prevent arbitrary file access."""

from __future__ import annotations

from pathlib import Path


def validate_path(path: str, base_directory: str | None = None) -> Path:
    """Validate that a file path is confined to a safe directory.

    Args:
        path: The path to validate.
        base_directory: The directory to confine access to.
            Defaults to the current working directory.

    Returns:
        The resolved, validated Path.

    Raises:
        ValueError: If the path escapes the base directory.
    """
    base = Path(base_directory).resolve() if base_directory else Path.cwd().resolve()
    resolved = (
        (base / path).resolve()
        if not Path(path).is_absolute()
        else Path(path).resolve()
    )

    if not resolved.is_relative_to(base):
        raise ValueError(
            f"Path {path!r} resolves to {resolved} which is outside "
            f"the allowed directory {base}."
        )

    return resolved
