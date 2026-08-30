"""Re-exports of version utilities from ``crewai_core.version``.

Kept as a stable import path for the framework; new code should import from
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


__all__ = [
    "check_version",
    "get_crewai_version",
    "get_latest_version_from_pypi",
    "is_current_version_yanked",
    "is_newer_version_available",
]
