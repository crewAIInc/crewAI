"""Checkpointable execution context for the crewAI runtime.

Captures the ContextVar state needed to resume execution from a checkpoint.
Used by the RootModel (step 5) to include execution context in snapshots.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.context import (
    _current_task_id,
    _platform_integration_token,
)
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


class ExecutionContext(BaseModel):
    """Snapshot of ContextVar state required for checkpoint/resume."""

    current_task_id: str | None = Field(default=None)
    flow_request_id: str | None = Field(default=None)
    flow_id: str | None = Field(default=None)
    flow_method_name: str = Field(default="unknown")

    event_id_stack: tuple[tuple[str, str], ...] = Field(default=())
    last_event_id: str | None = Field(default=None)
    triggering_event_id: str | None = Field(default=None)
    emission_sequence: int = Field(default=0)

    feedback_callback_info: dict[str, Any] | None = Field(default=None)
    platform_token: str | None = Field(default=None)


def capture_execution_context(
    feedback_callback_info: dict[str, Any] | None = None,
) -> ExecutionContext:
    """Read all checkpoint-required ContextVars into an ExecutionContext."""
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

    if ctx.platform_token is not None:
        _platform_integration_token.set(ctx.platform_token)
