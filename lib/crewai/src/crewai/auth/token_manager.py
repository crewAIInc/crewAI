"""Deprecated: use ``crewai_core.token_manager`` instead."""

from __future__ import annotations

import warnings

from crewai_core.token_manager import TokenManager as TokenManager


__all__ = ["TokenManager"]


warnings.warn(
    "crewai.auth.token_manager is deprecated; import from crewai_core.token_manager.",
    DeprecationWarning,
    stacklevel=2,
)
