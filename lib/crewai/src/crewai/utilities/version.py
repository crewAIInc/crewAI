"""Version utilities for crewAI."""

from __future__ import annotations

import importlib.metadata
from functools import cache


@cache
def get_crewai_version() -> str:
    """Get the installed crewAI version string."""
    return importlib.metadata.version("crewai")