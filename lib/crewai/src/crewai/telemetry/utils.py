"""Telemetry utility functions.

This module provides utility functions for telemetry operations.
"""

from __future__ import annotations

from collections.abc import Callable
import os
import sys
from typing import TYPE_CHECKING, Any, Final

from opentelemetry.trace import Span, Status, StatusCode


if TYPE_CHECKING:
    from crewai.crew import Crew
    from crewai.task import Task


# Environment variables set by AI coding assistants, checked in order.
# Only the assistant's name is ever recorded - never the variable's value.
_CODING_AGENT_ENV_MARKERS: Final[tuple[tuple[str, str], ...]] = (
    ("CLAUDECODE", "claude_code"),
    ("CLAUDE_CODE_ENTRYPOINT", "claude_code"),
    ("CURSOR_TRACE_ID", "cursor"),
    ("CURSOR_AGENT", "cursor"),
    ("CODEX_SANDBOX", "codex"),
    ("CODEX_SANDBOX_NETWORK_DISABLED", "codex"),
    ("GEMINI_CLI", "gemini_cli"),
    ("AIDER_MODEL", "aider"),
    ("WINDSURF_SESSION_ID", "windsurf"),
    ("DEVIN_SESSION_ID", "devin"),
    ("REPLIT_AGENT", "replit_agent"),
    ("COPILOT_AGENT_ID", "copilot"),
    ("GITHUB_COPILOT_CLI", "copilot"),
    ("OPENHANDS_SESSION_ID", "openhands"),
    ("CLINE_ACTIVE", "cline"),
    ("AMP_AGENT", "amp_code"),
)

# Editors whose integrated terminal implies a human is likely present. Used only
# as a weaker fallback when no explicit coding-agent marker is found.
_EDITOR_TERM_MARKERS: Final[tuple[tuple[str, str, str], ...]] = (
    ("TERM_PROGRAM", "vscode", "vscode_terminal"),
    ("TERMINAL_EMULATOR", "JetBrains-JediTerm", "jetbrains_terminal"),
)

_FALLBACK_AGENT_NAMES: Final[tuple[str, ...]] = ("non_interactive", "unknown")

# The complete set of values detect_coding_agent() can ever return. Every value
# is a literal defined in this module, which is what makes the function
# structurally incapable of emitting PII: no environment value, path, hostname,
# or user-supplied string can reach the return value.
KNOWN_CODING_AGENTS: Final[frozenset[str]] = frozenset(
    [name for _, name in _CODING_AGENT_ENV_MARKERS]
    + [name for _, _, name in _EDITOR_TERM_MARKERS]
    + list(_FALLBACK_AGENT_NAMES)
)


def detect_coding_agent() -> str:
    """Best-effort detection of the AI coding assistant running this process.

    Detection is based on environment variables that coding assistants set in
    the shells they spawn. Only the assistant's normalized name is returned -
    environment variable values are never read into the return value or
    recorded anywhere.

    This is intentionally heuristic: markers change as tools evolve, so a
    result of "unknown" means "no known marker present", not "no agent".

    Returns:
        A normalized assistant name (e.g. "claude_code", "cursor", "codex"),
        an editor terminal hint (e.g. "vscode_terminal"), "non_interactive"
        when no marker is found and there is no TTY, or "unknown" otherwise.
        The result is always a member of KNOWN_CODING_AGENTS.
    """
    for env_var, agent_name in _CODING_AGENT_ENV_MARKERS:
        if os.environ.get(env_var):
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
