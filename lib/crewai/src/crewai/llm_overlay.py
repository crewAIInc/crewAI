"""Process-context ``role -> model`` overlay for agents.

``llm_overlay`` lets a per-run caller make specific agents use a different
model without editing the code that builds them. While the block is active,
an ``Agent`` or ``LiteAgent`` whose ``role`` is a key of the mapping is built
with the mapped model instead of its declared ``llm``. Roles that are not
keys, and every agent built outside the block, keep their own model.

Roles are matched exactly, by the text they have when the overlay is read;
only whitespace around a role or a key is ignored, so a role a YAML file
leaves with a trailing newline matches a key written for the clean text.
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

A mapped model is built with the settings a caller configured on the
declared ``llm`` (``crewai.utilities.llm_utils.create_llm_like``): generation
and runtime settings — temperature, timeouts, token limits, stop sequences —
when the new model's class has the field and accepts the value; credentials,
endpoints and provider-specific settings only when the mapped model is on the
same provider — another provider gets its own defaults and environment. What a
provider derived from the model rather than took from the caller is derived
again for the new model.

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
            overlay for the block. Whitespace around each role is dropped from
            the copy the block uses, so a key matches a role that differs from
            it only by its surrounding whitespace; the mapping passed in is
            left as it is. The previous value is always restored on exit,
            including when the block raises.
    """
    token = active.set(_stripped(mapping))
    try:
        yield
    finally:
        active.reset(token)


def overlay_model_for(role: str | None) -> str | None:
    """The model the active overlay assigns to ``role``.

    Whitespace around ``role`` is ignored, as it was around the keys when the
    overlay was set. An empty or ``None`` role matches nothing.

    Returns:
        The mapped model string, or ``None`` when no overlay is active or
        ``role`` is not one of its keys.
    """
    mapping = active.get()
    if not mapping or role is None:
        return None
    role = role.strip()
    return mapping.get(role) if role else None


def _stripped(mapping: dict[str, str] | None) -> dict[str, str] | None:
    """A copy of ``mapping`` with the whitespace around each role dropped.

    Two keys that differ only by whitespace name one role. When they name the
    same model the copy holds it once; when they name different models the
    mapping is ambiguous and is refused, so the model an agent runs on never
    depends on dictionary order.
    """
    if mapping is None:
        return None
    stripped: dict[str, str] = {}
    for role, model in mapping.items():
        key = role.strip()
        if key in stripped and stripped[key] != model:
            raise ValueError(
                f"llm_overlay: role {key!r} is mapped twice with different models "
                f"({stripped[key]!r} and {model!r}); give each role one model"
            )
        stripped[key] = model
    return stripped
