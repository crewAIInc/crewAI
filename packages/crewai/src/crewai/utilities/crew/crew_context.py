"""Context management utilities for tracking crew and task execution context using OpenTelemetry baggage."""

from typing import Optional

from opentelemetry import baggage

from crewai.utilities.crew.models import CrewContext


def get_crew_context() -> Optional[CrewContext]:
    """Get the current crew context from OpenTelemetry baggage.

    Returns:
        CrewContext instance containing crew context information, or None if no context is set
    """
    return baggage.get_baggage("crew_context")
