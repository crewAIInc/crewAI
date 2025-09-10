"""Telemetry utility functions.

This module provides utility functions for telemetry operations.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Span, Status, StatusCode

if TYPE_CHECKING:
    from crewai.crew import Crew
    from crewai.task import Task


def add_agent_fingerprint_to_span(
    span: Span, agent: Any, add_attribute_fn: Callable[[Span, str, Any], None]
) -> None:
    """Add agent fingerprint data to a span if available.

    Args:
        span: The span to add the attributes to.
        agent: The agent whose fingerprint data should be added.
        add_attribute_fn: Function to add attributes to the span.
    """
    if agent:
        # Try to get fingerprint directly
        if hasattr(agent, "fingerprint") and agent.fingerprint:
            add_attribute_fn(span, "agent_fingerprint", agent.fingerprint.uuid_str)
            if hasattr(agent, "role"):
                add_attribute_fn(span, "agent_role", agent.role)
        else:
            # Try to get fingerprint using getattr (for cases where it might not be directly accessible)
            agent_fingerprint = getattr(
                getattr(agent, "fingerprint", None), "uuid_str", None
            )
            if agent_fingerprint:
                add_attribute_fn(span, "agent_fingerprint", agent_fingerprint)
                if hasattr(agent, "role"):
                    add_attribute_fn(span, "agent_role", agent.role)


def add_crew_attributes(
    span: Span,
    crew: "Crew",
    add_attribute_fn: Callable[[Span, str, Any], None],
    include_fingerprint: bool = True,
) -> None:
    """Add crew attributes to a span.

    Args:
        span: The span to add the attributes to.
        crew: The crew whose attributes should be added.
        add_attribute_fn: Function to add attributes to the span.
        include_fingerprint: Whether to include fingerprint data.
    """
    add_attribute_fn(span, "crew_key", crew.key)
    add_attribute_fn(span, "crew_id", str(crew.id))

    if include_fingerprint and hasattr(crew, "fingerprint") and crew.fingerprint:
        add_attribute_fn(span, "crew_fingerprint", crew.fingerprint.uuid_str)


def add_task_attributes(
    span: Span,
    task: "Task",
    add_attribute_fn: Callable[[Span, str, Any], None],
    include_fingerprint: bool = True,
) -> None:
    """Add task attributes to a span.

    Args:
        span: The span to add the attributes to.
        task: The task whose attributes should be added.
        add_attribute_fn: Function to add attributes to the span.
        include_fingerprint: Whether to include fingerprint data.
    """
    add_attribute_fn(span, "task_key", task.key)
    add_attribute_fn(span, "task_id", str(task.id))

    if include_fingerprint and hasattr(task, "fingerprint") and task.fingerprint:
        add_attribute_fn(span, "task_fingerprint", task.fingerprint.uuid_str)


def add_crew_and_task_attributes(
    span: Span,
    crew: "Crew",
    task: "Task",
    add_attribute_fn: Callable[[Span, str, Any], None],
    include_fingerprints: bool = True,
) -> None:
    """Add both crew and task attributes to a span.

    Args:
        span: The span to add the attributes to.
        crew: The crew whose attributes should be added.
        task: The task whose attributes should be added.
        add_attribute_fn: Function to add attributes to the span.
        include_fingerprints: Whether to include fingerprint data.
    """
    add_crew_attributes(span, crew, add_attribute_fn, include_fingerprints)
    add_task_attributes(span, task, add_attribute_fn, include_fingerprints)


def close_span(span: Span) -> None:
    """Set span status to OK and end it.

    Args:
        span: The span to close.
    """
    span.set_status(Status(StatusCode.OK))
    span.end()
