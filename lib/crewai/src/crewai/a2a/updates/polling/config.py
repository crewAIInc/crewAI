"""Backward-compatibility shim — use ``crewai_a2a.updates.polling.config`` instead."""

import warnings


warnings.warn(
    "'crewai.a2a.updates.polling.config' has been moved to 'crewai_a2a.updates.polling.config'. "
    "Please update your imports. The old path will be removed in v2.0.0.",
    FutureWarning,
    stacklevel=2,
)

from crewai_a2a.updates.polling.config import *  # noqa: E402, F403
