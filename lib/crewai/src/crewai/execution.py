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


def clear_execution_uuid(token: contextvars.Token[str | None]) -> None:
    """Restore the execution uuid that was active before ``token`` was issued.

    ``token`` is required so this cannot wipe an outer kickoff's uuid.
    """
    _current_execution_uuid.reset(token)


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
    """End an execution context owned by the current kickoff.

    Nested kickoffs pass ``None`` and leave the outer uuid in place.
    """
    if token is not None:
        clear_execution_uuid(token)
