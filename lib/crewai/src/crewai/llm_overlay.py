"""Process-context ``role -> model`` overlay for agents.

``llm_overlay`` lets a per-run caller make specific agents use a different
model without editing the code that builds them. While the block is active,
an ``Agent`` or ``LiteAgent`` whose ``role`` is a key of the mapping is built
with the mapped model instead of its declared ``llm``. Roles that are not
keys, and every agent built outside the block, keep their own model.

The overlay is a :class:`contextvars.ContextVar`, so it follows the calling
context, not the process. A plain ``threading.Thread`` started inside the
block does not see it: callers that run agents in their own threads must
propagate the context themselves, e.g.
``contextvars.copy_context().run(build_and_run_agents)``.

Example:
    >>> with llm_overlay({"Researcher": "openai/gpt-4o"}):
    ...     crew = build_crew()  # the Researcher agent is built on gpt-4o
    ...     crew.kickoff()
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
