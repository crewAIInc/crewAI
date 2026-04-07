"""Compatibility shim — re-exports from crewai_tools.security.safe_path.

Import from crewai_tools.security.safe_path instead.
"""

from crewai_tools.security.safe_path import (
    validate_directory_path,
    validate_file_path,
    validate_url,
)


__all__ = ["validate_directory_path", "validate_file_path", "validate_url"]
