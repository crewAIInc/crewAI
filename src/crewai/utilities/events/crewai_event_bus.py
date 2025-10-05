"""Backwards compatibility stub for crewai.utilities.events.crewai_event_bus."""

import warnings

from crewai.events import crewai_event_bus

warnings.warn(
    "Importing from 'crewai.utilities.events.crewai_event_bus' is deprecated and will be removed in v1.0.0. "
    "Please use 'from crewai.events import crewai_event_bus' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["crewai_event_bus"]
