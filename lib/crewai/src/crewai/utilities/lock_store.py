"""Deprecated: use ``crewai_core.lock_store`` instead."""

from __future__ import annotations

import warnings

from crewai_core.lock_store import lock as lock


__all__ = ["lock"]


warnings.warn(
    "crewai.utilities.lock_store is deprecated; import from crewai_core.lock_store.",
    DeprecationWarning,
    stacklevel=2,
)
