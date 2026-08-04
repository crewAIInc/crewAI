"""Telemetry utility functions.

This module provides utility functions for telemetry operations.
"""

from __future__ import annotations

from collections.abc import Callable
import os
import sys
from typing import TYPE_CHECKING, Any, Final

from opentelemetry.trace import Span, Status, StatusCode

from crewai.utilities.constants import CODING_AGENT_ENV_MARKERS


if TYPE_CHECKING:
    from crewai.crew import Crew
    from crewai.task import Task


# Editors whose integrated terminal implies a human is likely present. Used only
# as a weaker fallback when no explicit coding-agent marker is found.
_EDITOR_TERM_MARKERS: Final[tuple[tuple[str, str, str], ...]] = (
    ("TERM_PROGRAM", "vscode", "vscode_terminal"),
    ("TERMINAL_EMULATOR", "JetBrains-JediTerm", "jetbrains_terminal"),
)

_FALLBACK_AGENT_NAMES: Final[tuple[str, ...]] = ("non_interactive", "unknown")

# The complete set of values detect_coding_agent() can ever return. Every value
# is a literal from CODING_AGENT_ENV_MARKERS or this module, which is what makes
# the function structurally incapable of emitting PII: no environment value,
# path, hostname, or user-supplied string can reach the return value.
KNOWN_CODING_AGENTS: Final[frozenset[str]] = frozenset(
    [name for name, _ in CODING_AGENT_ENV_MARKERS]
    + [name for _, _, name in _EDITOR_TERM_MARKERS]
    + list(_FALLBACK_AGENT_NAMES)
)


def detect_coding_agent() -> str:
    """Best-effort detection of the AI coding assistant running this process.

    Uses the shared ``CODING_AGENT_ENV_MARKERS`` table, so this agrees with the
    env-context events emitted by ``get_env_context()`` rather than maintaining
    a second, narrower set of markers. Precedence follows that table: Claude
    Code, then Codex, then Cursor, then the remaining assistants.

    Only the assistant's normalized name is returned - environment variable
    values are never read into the return value or recorded anywhere.

    Two limits worth knowing. This is heuristic: markers change as tools
    evolve, so "unknown" means "no known marker present", not "no agent". And
    some markers (the Cursor set in particular) are set by the editor for any
    integrated terminal, so a result names the environment the process is
    running *under*, not proof that an agent authored the code.

    Returns:
        A normalized assistant name (e.g. "claude_code", "cursor", "codex"),
        an editor terminal hint (e.g. "vscode_terminal"), "non_interactive"
        when no marker is found and there is no TTY, or "unknown" otherwise.
        The result is always a member of KNOWN_CODING_AGENTS.
    """
    for agent_name, env_vars in CODING_AGENT_ENV_MARKERS:
        if any(os.environ.get(env_var) for env_var in env_vars):
            return agent_name

    for env_var, expected, agent_name in _EDITOR_TERM_MARKERS:
        if os.environ.get(env_var) == expected:
            return agent_name

    try:
        if not sys.stdout.isatty():
            return "non_interactive"
    except (AttributeError, ValueError, OSError):
        return "unknown"

    return "unknown"


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
        if hasattr(agent, "fingerprint") and agent.fingerprint:
            add_attribute_fn(span, "agent_fingerprint", agent.fingerprint.uuid_str)
            if hasattr(agent, "role"):
                add_attribute_fn(span, "agent_role", agent.role)
        else:
            agent_fingerprint = getattr(
                getattr(agent, "fingerprint", None), "uuid_str", None
            )
            if agent_fingerprint:
                add_attribute_fn(span, "agent_fingerprint", agent_fingerprint)
                if hasattr(agent, "role"):
                    add_attribute_fn(span, "agent_role", agent.role)


def add_crew_attributes(
    span: Span,
    crew: Crew,
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
    task: Task,
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
    crew: Crew,
    task: Task,
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
