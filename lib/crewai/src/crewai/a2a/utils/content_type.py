"""Backward-compatibility shim — use ``crewai_a2a.utils.content_type`` instead."""

import warnings


warnings.warn(
    "'crewai.a2a.utils.content_type' has been moved to 'crewai_a2a.utils.content_type'. "
    "Please update your imports. The old path will be removed in v2.0.0.",
    FutureWarning,
    stacklevel=2,
)

from crewai_a2a.utils.content_type import *  # noqa: E402, F403
