"""Re-exports of version utilities from ``crewai_core.version``.

Kept as a stable import path for the CLI; new code should import from
``crewai_core.version`` directly.
"""

from __future__ import annotations

from crewai_core.version import (
    check_version as check_version,
    get_crewai_version as get_crewai_version,
    get_latest_version_from_pypi as get_latest_version_from_pypi,
    is_current_version_yanked as is_current_version_yanked,
    is_newer_version_available as is_newer_version_available,
)
from packaging.version import Version

from crewai_cli import __version__ as _crewai_cli_version


def get_crewai_dependency_range(current_version: str | None = None) -> str:
    """Return the supported CrewAI dependency range for generated projects."""
    parsed_version = Version(current_version or _crewai_cli_version)
    return f">={parsed_version},<{parsed_version.major + 1}.0.0"


def get_crewai_tools_dependency(current_version: str | None = None) -> str:
    """Return the generated-project dependency for CrewAI with tools."""
    return f"crewai[tools]{get_crewai_dependency_range(current_version)}"


__all__ = [
    "check_version",
    "get_crewai_dependency_range",
    "get_crewai_tools_dependency",
    "get_crewai_version",
    "get_latest_version_from_pypi",
    "is_current_version_yanked",
    "is_newer_version_available",
]
