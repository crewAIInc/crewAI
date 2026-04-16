"""Version utilities for crewAI."""

from __future__ import annotations

from functools import cache
import importlib.metadata


@cache
def get_crewai_version() -> str:
    """Get the installed crewAI version string."""
    return importlib.metadata.version("crewai")
