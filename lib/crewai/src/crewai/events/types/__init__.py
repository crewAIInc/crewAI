"""Event type definitions for CrewAI.

This module contains all event types used throughout the CrewAI system
for monitoring and extending agent, crew, task, and tool execution.
"""

from crewai.events.types.optimizer_events import (
    OptimizationCompletedEvent,
    OptimizationFailedEvent,
    OptimizationStartedEvent,
)


# OptimizationTrialCompletedEvent is intentionally excluded from __all__:
# DSPy teleprompters do not expose a per-trial callback, so the event cannot
# be emitted accurately. The class is defined in optimizer_events.py and will
# be added here when DSPy adds callback support.

__all__ = [
    "OptimizationCompletedEvent",
    "OptimizationFailedEvent",
    "OptimizationStartedEvent",
]
