"""Typed contexts for the interception points wired in phases 2-5.

Each context is a dataclass whose fields are nullable and defaulted, so a field
that is not meaningful for a given runtime (e.g. ``agent_role`` inside a flow)
is simply ``None`` rather than an error. Every context exposes a ``payload``
field: the interceptable value a hook may mutate in place or replace by
returning a new value.

The legacy ``pre/post_model_call`` and ``pre/post_tool_call`` points keep using
:class:`~crewai.hooks.llm_hooks.LLMCallHookContext` and
:class:`~crewai.hooks.tool_hooks.ToolCallHookContext` for backwards
compatibility; they are intentionally not redefined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class InterceptionContext:
    """Base context shared by the framework-native interception points."""

    payload: Any = None
    agent: Any = None
    agent_role: str | None = None
    task: Any = None
    crew: Any = None
    flow: Any = None


@dataclass
class ExecutionStartContext(InterceptionContext):
    """``execution_start``: a crew or flow is about to begin. ``payload`` = inputs."""

    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputContext(InterceptionContext):
    """``input``: resolved inputs for an execution. ``payload`` = inputs."""

    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputContext(InterceptionContext):
    """``output``: final result of a crew or flow. ``payload`` = the output object."""

    output: Any = None


@dataclass
class ExecutionEndContext(InterceptionContext):
    """``execution_end``: a crew or flow has finished. ``payload`` = the output object.

    Dispatched on both successful and failed executions. ``status`` is
    ``"completed"`` on success and ``"failed"`` when the execution raised;
    ``error`` carries the exception on failure (``output``/``payload`` are
    ``None`` in that case).
    """

    output: Any = None
    status: Literal["completed", "failed"] = "completed"
    error: BaseException | None = None


@dataclass
class StepContext(InterceptionContext):
    """``pre_step`` / ``post_step``: a task or flow-method step boundary.

    ``kind`` is ``"task"`` for crew tasks and ``"flow_method"`` for flow methods.
    ``payload`` is the step input (pre) or step output (post).
    """

    kind: str | None = None
    step_name: str | None = None
    output: Any = None
