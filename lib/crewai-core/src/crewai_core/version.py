"""Version utilities for CrewAI."""

from __future__ import annotations

from functools import cache
import importlib.metadata


@cache
def get_crewai_version() -> str:
    """Return the installed crewAI version string.

    Falls back to ``"unknown"`` when neither crewai nor crewai-core are
    pip-installed (e.g. running directly from a source checkout).
    """
    try:
        return importlib.metadata.version("crewai")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        return importlib.metadata.version("crewai-core")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
