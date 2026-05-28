"""Experimental feature gate for the Skills Repository."""

from __future__ import annotations

import os


ENV_VAR = "CREWAI_EXPERIMENTAL"


class ExperimentalFeatureDisabledError(RuntimeError):
    """Raised when an experimental feature is used without the flag set."""


def is_enabled() -> bool:
    return os.environ.get(ENV_VAR) == "1"


def require_experimental_skills() -> None:
    if not is_enabled():
        raise ExperimentalFeatureDisabledError(
            "The Skills Repository (registry refs, cache, downloads) is "
            f"experimental. Set {ENV_VAR}=1 to enable it."
        )
