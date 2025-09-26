"""Backwards compatibility stub for crewai.utilities.events.base_event_listener."""

import warnings

from crewai.events import BaseEventListener

warnings.warn(
    "Importing from 'crewai.utilities.events.base_event_listener' is deprecated and will be removed in v1.0.0. "
    "Please use 'from crewai.events import BaseEventListener' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BaseEventListener"]
