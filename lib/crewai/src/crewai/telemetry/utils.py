"""Telemetry utility functions.

This module provides utility functions for telemetry operations.
"""

from __future__ import annotations

from collections.abc import Callable
import os
import sys
from typing import TYPE_CHECKING, Any, Final

from opentelemetry.trace import Span, Status, StatusCode

from crewai.utilities.constants import (
    CODING_AGENT_ENV_MARKERS,
    GENERIC_AGENT_ENV_VARS,
    RUNTIME_CONTEXT_ENV_MARKERS,
)


if TYPE_CHECKING:
    from crewai.crew import Crew
    from crewai.task import Task


# Editors whose integrated terminal a process can be launched from. Matched on
# an exact value rather than presence, since these variables name the terminal.
_EDITOR_TERM_MARKERS: Final[tuple[tuple[str, str, str], ...]] = (
    ("TERM_PROGRAM", "vscode", "vscode_terminal"),
    ("TERMINAL_EMULATOR", "JetBrains-JediTerm", "jetbrains_terminal"),
)

# Marks a process running inside a container when no more specific runtime
# marker is present.
_DOCKER_ENV_PATH: Final[str] = "/.dockerenv"

_UNKNOWN: Final[str] = "unknown"
_OTHER: Final[str] = "other"

# The complete set of values detect_coding_agent() can ever return. Every value
# is a literal from CODING_AGENT_ENV_MARKERS or this module, which is what makes
# the function structurally incapable of emitting PII: no environment value,
# path, hostname, or user-supplied string can reach the return value.
KNOWN_CODING_AGENTS: Final[frozenset[str]] = frozenset(
    [name for name, _ in CODING_AGENT_ENV_MARKERS] + [_OTHER, _UNKNOWN]
)

# The same guarantee for detect_runtime_context(): a closed set of literals.
KNOWN_RUNTIME_CONTEXTS: Final[frozenset[str]] = frozenset(
    [name for name, _ in RUNTIME_CONTEXT_ENV_MARKERS]
    + [name for _, _, name in _EDITOR_TERM_MARKERS]
    + ["interactive", "non_interactive", _UNKNOWN]
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

    Answers only *which assistant*. Where the process runs is
    :func:`detect_runtime_context`, so an "unknown" here is a genuine gap in
    the marker table rather than a run that never had an assistant to find.

    Returns:
        A normalized assistant name (e.g. "claude_code", "cursor", "codex"), or
        "unknown" when no marker matches. The result is always a member of
        KNOWN_CODING_AGENTS.
    """
    for agent_name, env_vars in CODING_AGENT_ENV_MARKERS:
        if any(os.environ.get(env_var) for env_var in env_vars):
            return agent_name

    # Checked last and by presence: the cross-vendor marker establishes that an
    # assistant is present without naming one, and an empty value still says so.
    if any(env_var in os.environ for env_var in GENERIC_AGENT_ENV_VARS):
        return _OTHER

    return _UNKNOWN


def detect_runtime_context() -> str:
    """Best-effort detection of where this process is running.

    Separate from :func:`detect_coding_agent` because the two answer different
    questions. Automated runs -- CI, containers, serverless -- have no
    assistant to detect, and reporting them in the assistant field made an
    unrecognized assistant indistinguishable from a run that could never have
    had one.

    Precedence runs most specific first: CI and hosted IDEs typically run
    inside containers, so a bare container match only applies once the more
    specific markers have been ruled out.

    Returns:
        One of the runtime names (e.g. "ci", "serverless", "container",
        "notebook", "hosted_ide"), an editor terminal (e.g. "vscode_terminal"),
        "interactive" or "non_interactive" when only a TTY check applies, or
        "unknown" when that check cannot be made. The result is always a member
        of KNOWN_RUNTIME_CONTEXTS.
    """
    # Presence, not truthiness: a platform that exports an empty CI= is still
    # CI, unlike the assistant markers where an empty value means the tool set
    # a placeholder rather than claiming the session.
    for context_name, env_vars in RUNTIME_CONTEXT_ENV_MARKERS:
        if any(env_var in os.environ for env_var in env_vars):
            return context_name

    for env_var, expected, context_name in _EDITOR_TERM_MARKERS:
        if os.environ.get(env_var) == expected:
            return context_name

    if os.path.exists(_DOCKER_ENV_PATH):
        return "container"

    try:
        return "interactive" if sys.stdout.isatty() else "non_interactive"
    except (AttributeError, ValueError, OSError):
        return _UNKNOWN


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
