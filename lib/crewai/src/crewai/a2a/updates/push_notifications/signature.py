"""Backward-compatibility shim — use ``crewai_a2a.updates.push_notifications.signature`` instead."""

import warnings


warnings.warn(
    "'crewai.a2a.updates.push_notifications.signature' has been moved to 'crewai_a2a.updates.push_notifications.signature'. "
    "Please update your imports. The old path will be removed in v2.0.0.",
    FutureWarning,
    stacklevel=2,
)

from crewai_a2a.updates.push_notifications.signature import *  # noqa: E402, F403
