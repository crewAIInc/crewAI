"""Deprecated: use ``crewai_core.version`` instead."""

from __future__ import annotations

import warnings

from crewai_core.version import get_crewai_version as get_crewai_version


__all__ = ["get_crewai_version"]


warnings.warn(
    "crewai.utilities.version is deprecated; import from crewai_core.version.",
    DeprecationWarning,
    stacklevel=2,
)
