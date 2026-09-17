"""Process-context ``role -> model`` overlay for agents.

``llm_overlay`` lets a per-run caller make specific agents use a different
model without editing the code that builds them. While the block is active,
an ``Agent`` or ``LiteAgent`` whose ``role`` is a key of the mapping is built
with the mapped model instead of its declared ``llm``. Roles that are not
keys, and every agent built outside the block, keep their own model.

Roles are matched exactly, by the text they have when the overlay is read.
An ``Agent`` is read when it is built, with its declared role, and read
again when ``crew.kickoff(inputs=...)`` interpolates the inputs into that
role and the text changes. The second read is what lets a role declared as
a template (``"Researcher for {repo}"``, the YAML form of a ``CrewBase``
crew) match a key written for the interpolated text, which is the role
every trace records; the template itself is not a key at construction. A
role the interpolation leaves unchanged is not read again. Inside a block,
an agent runs on the model its current role maps to, and on its declared
``llm`` when that role is not a key: a kickoff that interpolates to a role
outside the mapping puts the declared ``llm`` instance back, so a ``Crew``
reused across kickoffs with different inputs never bills a previous role's
provider. Outside any block an interpolation changes nothing.

A mapped model is built with the declared ``llm``'s configuration
(``crewai.utilities.llm_utils.create_llm_like``): temperature, timeouts,
token limits, stop sequences and the like always; credentials, endpoints and
provider-specific settings only when the mapped model is on the same
provider — another provider gets its own defaults and environment.

The overlay is a :class:`contextvars.ContextVar`, so it follows the calling
context, not the process. A plain ``threading.Thread`` started inside the
block does not see it: callers that run agents in their own threads must
propagate the context themselves, e.g.
``contextvars.copy_context().run(build_and_run_agents)``. With
``Crew(stream=True)``, ``kickoff`` returns a stream and runs the crew — and the
kickoff-time read — when that stream is first iterated, in a copy of the
context taken then: iterate it inside the block.

Each ``Agent`` reads the overlay once when built and once more only if a
kickoff rewrites its role; a re-validation of an existing agent (the event bus
registers an agent when it first emits) does not read it again, so an agent
built outside a block keeps its model through a kickoff inside one.

Example:
    >>> with llm_overlay({"Researcher": "openai/gpt-4o"}):
    ...     crew = build_crew()  # the Researcher agent is built on gpt-4o
    ...     crew.kickoff()

    >>> with llm_overlay({"Researcher for crewAIInc/x": "openai/gpt-4o"}):
    ...     crew = build_crew()  # role "Researcher for {repo}": declared llm
    ...     crew.kickoff(inputs={"repo": "crewAIInc/x"})  # now runs on gpt-4o
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


active: ContextVar[dict[str, str] | None] = ContextVar(
    "llm_overlay_active", default=None
)


@contextmanager
def llm_overlay(mapping: dict[str, str] | None) -> Iterator[None]:
    """Route agent roles to models for the duration of the block.

    Args:
        mapping: ``{role: model}`` for the block; ``None`` clears any active
            overlay for the block. The previous value is always restored on
            exit, including when the block raises.
    """
    token = active.set(mapping)
    try:
        yield
    finally:
        active.reset(token)


def overlay_model_for(role: str) -> str | None:
    """The model the active overlay assigns to ``role``.

    Returns:
        The mapped model string, or ``None`` when no overlay is active or
        ``role`` is not one of its keys.
    """
    mapping = active.get()
    return mapping.get(role) if mapping else None
