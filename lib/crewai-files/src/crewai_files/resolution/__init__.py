"""File resolution logic."""

from crewai_files.resolution.resolver import FileResolver
from crewai_files.resolution.utils import (
    is_file_source,
    normalize_input_files,
    wrap_file_source,
)


__all__ = [
    "FileResolver",
    "is_file_source",
    "normalize_input_files",
    "wrap_file_source",
]
