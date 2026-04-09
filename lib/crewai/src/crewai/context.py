from collections.abc import Generator
from contextlib import contextmanager
import contextvars
import os
from typing import Any

from pydantic import BaseModel, Field

from crewai.events.base_events import (
    get_emission_sequence,
    set_emission_counter,
)
from crewai.events.event_context import (
    _event_id_stack,
    _last_event_id,
    _triggering_event_id,
)
from crewai.flow.flow_context import (
    current_flow_id,
    current_flow_method_name,
    current_flow_request_id,
)


_platform_integration_token: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("platform_integration_token", default=None)
)


def set_platform_integration_token(integration_token: str) -> None:
    """Set the platform integration token in the current context.

    Args:
        integration_token: The integration token to set.
    """
    _platform_integration_token.set(integration_token)


def get_platform_integration_token() -> str | None:
    """Get the platform integration token from the current context or environment.

    Returns:
        The integration token if set, otherwise None.
    """
    token = _platform_integration_token.get()
    if token is None:
        token = os.getenv("CREWAI_PLATFORM_INTEGRATION_TOKEN")
    return token


@contextmanager
def platform_context(integration_token: str) -> Generator[None, Any, None]:
    """Context manager to temporarily set the platform integration token.

    Args:
      integration_token: The integration token to set within the context.
    """
    token = _platform_integration_token.set(integration_token)
    try:
        yield
    finally:
        _platform_integration_token.reset(token)


_current_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_task_id", default=None
)


def set_current_task_id(task_id: str | None) -> contextvars.Token[str | None]:
    """Set the current task ID in the context. Returns a token for reset."""
    return _current_task_id.set(task_id)


def reset_current_task_id(token: contextvars.Token[str | None]) -> None:
    """Reset the current task ID to its previous value."""
    _current_task_id.reset(token)


def get_current_task_id() -> str | None:
    """Get the current task ID from the context."""
    return _current_task_id.get()


class ExecutionContext(BaseModel):
    """Snapshot of ContextVar execution state."""

    current_task_id: str | None = Field(default=None)
    flow_request_id: str | None = Field(default=None)
    flow_id: str | None = Field(default=None)
    flow_method_name: str = Field(default="unknown")

    event_id_stack: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    last_event_id: str | None = Field(default=None)
    triggering_event_id: str | None = Field(default=None)
    emission_sequence: int = Field(default=0)

    feedback_callback_info: dict[str, Any] | None = Field(default=None)
    platform_token: str | None = Field(default=None)


def capture_execution_context(
    feedback_callback_info: dict[str, Any] | None = None,
) -> ExecutionContext:
    """Read current ContextVars into an ExecutionContext."""
    return ExecutionContext(
        current_task_id=_current_task_id.get(),
        flow_request_id=current_flow_request_id.get(),
        flow_id=current_flow_id.get(),
        flow_method_name=current_flow_method_name.get(),
        event_id_stack=_event_id_stack.get(),
        last_event_id=_last_event_id.get(),
        triggering_event_id=_triggering_event_id.get(),
        emission_sequence=get_emission_sequence(),
        feedback_callback_info=feedback_callback_info,
        platform_token=_platform_integration_token.get(),
    )


def apply_execution_context(ctx: ExecutionContext) -> None:
    """Write an ExecutionContext back into the ContextVars."""
    _current_task_id.set(ctx.current_task_id)
    current_flow_request_id.set(ctx.flow_request_id)
    current_flow_id.set(ctx.flow_id)
    current_flow_method_name.set(ctx.flow_method_name)

    _event_id_stack.set(ctx.event_id_stack)
    _last_event_id.set(ctx.last_event_id)
    _triggering_event_id.set(ctx.triggering_event_id)
    set_emission_counter(ctx.emission_sequence)

    _platform_integration_token.set(ctx.platform_token)
