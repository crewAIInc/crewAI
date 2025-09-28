"""Context management utilities for tracking crew and task execution context using OpenTelemetry baggage."""

from typing import cast

from opentelemetry import baggage

from crewai.utilities.crew.models import CrewContext


def get_crew_context() -> CrewContext | None:
    """Get the current crew context from OpenTelemetry baggage.

    Returns:
        CrewContext instance containing crew context information, or None if no context is set
    """
    return cast(CrewContext | None, baggage.get_baggage("crew_context"))
