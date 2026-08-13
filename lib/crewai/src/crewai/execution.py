"""Per-run execution identity for OSS traces.

Wharf keys spans by ``crewai.execution_uuid``. Enterprise stamps that from the
Celery ``kickoff_id`` inside ``telemetry_session``. Standalone OSS has no such
session, so crew/flow ``kickoff`` creates a uuid for the **outermost** run and
nested kickoffs (crew-in-flow, AgentExecutor, child flows) inherit it via
contextvars on the execution thread.

Minting lives on the kickoff call path (not the event bus): bus handlers run
on worker threads and cannot publish contextvars back to the user thread.

Enterprise (or any host) can call :func:`set_execution_uuid` before kickoff;
:func:`begin_execution` will not overwrite it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import contextvars
from uuid import uuid4


_current_execution_uuid: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "crewai_execution_uuid", default=None
)


def get_execution_uuid() -> str | None:
    """Return the active execution uuid, if any."""
    return _current_execution_uuid.get()


def set_execution_uuid(execution_uuid: str) -> contextvars.Token[str | None]:
    """Bind an execution uuid for the current context.

    Use this from enterprise / hosts that already own the run id (e.g. Celery
    ``kickoff_id``). Overwrites any previously bound value.
    """
    if not execution_uuid:
        raise ValueError("execution_uuid must be a non-empty string")
    return _current_execution_uuid.set(execution_uuid)


def ensure_execution_uuid(preferred: str | None = None) -> str:
    """Return the active execution uuid, creating one if needed.

    Never overwrites an existing value — nested kickoffs and enterprise
    sessions inherit the outer id.
    """
    existing = _current_execution_uuid.get()
    if existing is not None:
        return existing
    value = preferred or str(uuid4())
    if not value:
        raise ValueError("execution_uuid must be a non-empty string")
    _current_execution_uuid.set(value)
    return value


def clear_execution_uuid(token: contextvars.Token[str | None] | None = None) -> None:
    """Reset the execution uuid.

    Prefer passing the :class:`~contextvars.Token` returned when the value was
    set so nested contexts restore correctly. Without a token, clears to
    ``None``.
    """
    if token is not None:
        _current_execution_uuid.reset(token)
    else:
        _current_execution_uuid.set(None)


@contextmanager
def execution_uuid_scope(
    execution_uuid: str | None = None, *, force: bool = False
) -> Iterator[str]:
    """Bind an execution uuid for the duration of the block.

    Args:
        execution_uuid: Value to bind. When ``None`` and ``force`` is false,
            reuses the active uuid or creates a new one.
        force: When true, ``execution_uuid`` must be provided and replaces any
            active value (enterprise / host path).
    """
    if force:
        if not execution_uuid:
            raise ValueError("execution_uuid is required when force=True")
        token = set_execution_uuid(execution_uuid)
        try:
            yield execution_uuid
        finally:
            clear_execution_uuid(token)
        return

    existing = _current_execution_uuid.get()
    if existing is not None:
        yield existing
        return

    value = execution_uuid or str(uuid4())
    token = set_execution_uuid(value)
    try:
        yield value
    finally:
        clear_execution_uuid(token)


def begin_execution(
    execution_uuid: str | None = None,
) -> contextvars.Token[str | None] | None:
    """Start an execution context unless one is already active.

    The outermost crew or flow receives a new uuid by default. Nested
    executions inherit the active value and return no reset token.
    """
    if _current_execution_uuid.get() is not None:
        return None
    return set_execution_uuid(execution_uuid or str(uuid4()))


def end_execution(token: contextvars.Token[str | None] | None) -> None:
    """End an execution context owned by the current kickoff."""
    if token is not None:
        clear_execution_uuid(token)
