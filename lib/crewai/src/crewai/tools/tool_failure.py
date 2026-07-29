"""Structured signalling for tools that run but do not succeed.

A tool can complete without raising and still fail: Slack answers ``HTTP 200``
with ``{"ok": false, ...}``, an MCP server sets ``isError``. The call
"worked", so the error used to reach the agent as an ordinary string and the
run was recorded as a success.

A tool declares failure by returning a :class:`ToolFailure`; the policy
(:class:`ToolFailurePolicy`) decides the reaction. Detection is strictly
declarative -- nothing here guesses whether a string "looks like" an error.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.crew import Crew
    from crewai.lite_agent import LiteAgent
    from crewai.task import Task


class ToolFailureReason(str, Enum):
    """Why a tool call is considered unsuccessful."""

    TOOL_REPORTED = "tool_reported"
    """The tool itself returned a :class:`ToolFailure`."""

    EXCEPTION = "exception"
    """The tool raised; the framework caught it and fed the text to the agent."""

    MCP_ERROR = "mcp_error"
    """An MCP server answered with ``isError: true``."""

    USAGE_LIMIT = "usage_limit"
    """The tool's ``max_usage_count`` was already spent."""

    BLOCKED_BY_HOOK = "blocked_by_hook"
    """A ``before_tool_call`` hook refused the call."""

    UNKNOWN_TOOL = "unknown_tool"
    """The agent asked for a tool that does not exist."""

    INVALID_INPUT = "invalid_input"
    """Arguments could not be parsed or validated into the tool's schema."""


class ToolFailurePolicy(str, Enum):
    """How an agent reacts when one of its tools reports a failure."""

    IGNORE = "ignore"
    """Pre-1.16 behavior: the failure is not recorded, emitted, or acted on."""

    WARN = "warn"
    """Record the failure, emit an event, and keep going. The default."""

    RAISE = "raise"
    """Record the failure, emit an event, then abort with
    :class:`ToolExecutionFailedError`."""


class ToolFailure(BaseModel):
    """A tool's own report that it did not do what it was asked.

    Return one from ``_run``/``_arun`` instead of an error string. The agent
    still sees text via :meth:`as_agent_message`, so model behavior is
    unchanged -- but the framework now knows the call failed.
    """

    model_config = ConfigDict(frozen=True)

    message: str = Field(
        description="Human and LLM readable explanation of what went wrong."
    )
    reason: ToolFailureReason = Field(
        default=ToolFailureReason.TOOL_REPORTED,
        description="Category of failure, for grouping and filtering.",
    )
    code: str | None = Field(
        default=None,
        description=(
            "Machine-readable identifier from the failing system, "
            "e.g. 'channel_not_found'."
        ),
    )
    retryable: bool = Field(
        default=False,
        description="Whether retrying the same call could plausibly succeed.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra structured context the tool wants to preserve.",
    )

    def as_agent_message(self) -> str:
        """Render the text the agent sees for this failure."""
        if self.code:
            return f"{self.message} (code: {self.code})"
        return self.message


class ToolFailureRecord(BaseModel):
    """A :class:`ToolFailure` plus the context of the call that produced it.

    Lands on ``TaskOutput.tool_failures`` and on the event bus, so consumers
    never parse a string to learn that a step failed.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(description="Name of the tool that failed.")
    failure: ToolFailure = Field(description="The failure the tool reported.")
    tool_args: dict[str, Any] | str | None = Field(
        default=None, description="Arguments the tool was called with."
    )
    agent_role: str | None = Field(
        default=None, description="Role of the agent that made the call."
    )
    task_name: str | None = Field(
        default=None, description="Name or description of the task in flight."
    )
    task_id: str | None = Field(default=None, description="Id of the task in flight.")

    @property
    def message(self) -> str:
        """Shorthand for the underlying failure message."""
        return self.failure.message

    def summary(self) -> str:
        """One-line description suitable for logs and error messages."""
        where = f" during '{self.task_name}'" if self.task_name else ""
        return (
            f"Tool '{self.tool_name}' failed{where}: {self.failure.as_agent_message()}"
        )


class ToolExecutionFailedError(Exception):
    """Raised when a tool reports failure under :attr:`ToolFailurePolicy.RAISE`."""

    def __init__(self, record: ToolFailureRecord) -> None:
        self.record = record
        super().__init__(record.summary())


def detect_tool_failure(result: Any) -> ToolFailure | None:
    """Return the failure a tool declared, if it declared one.

    Only an explicit :class:`ToolFailure` counts, so a tool legitimately
    returning text about an error is never misread as having failed.
    """
    if isinstance(result, ToolFailure):
        return result
    return None


def failure_from_exception(
    error: BaseException, *, retryable: bool = False
) -> ToolFailure:
    """Build a :class:`ToolFailure` from an exception a tool raised."""
    return ToolFailure(
        message=str(error) or error.__class__.__name__,
        reason=ToolFailureReason.EXCEPTION,
        code=error.__class__.__name__,
        retryable=retryable,
    )


def resolve_tool_failure_policy(
    tool: Any = None,
    agent: BaseAgent | LiteAgent | None = None,
    task: Task | None = None,
    crew: Crew | None = None,
) -> ToolFailurePolicy:
    """Resolve the effective policy for one call.

    Most specific wins: tool, task, agent, crew, then
    :attr:`ToolFailurePolicy.WARN`. Callers pass either a ``BaseTool`` or the
    ``CrewStructuredTool`` wrapping it, so both are read -- otherwise a
    tool-scoped policy is ignored on every native function-calling path.
    """
    original_tool = getattr(tool, "_original_tool", None) if tool is not None else None

    for source in (tool, original_tool, task, agent, crew):
        if source is None:
            continue
        policy = getattr(source, "tool_failure_policy", None)
        if policy is None:
            continue
        try:
            return ToolFailurePolicy(policy)
        except ValueError:
            # A malformed policy must not take down a tool call.
            logger.warning(
                "Ignoring invalid tool_failure_policy %r on %s; expected one of %s.",
                policy,
                type(source).__name__,
                [member.value for member in ToolFailurePolicy],
            )
    return ToolFailurePolicy.WARN


def collect_tool_failures(agent: Any) -> list[ToolFailureRecord]:
    """Return the failures recorded on an agent, tolerating custom agents.

    Third-party agents and test doubles may not expose ``last_tool_failures``
    as a list, and building a task's output must never fail over telemetry.
    """
    records = getattr(agent, "last_tool_failures", None)
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, ToolFailureRecord)]


def _record_on_agent(agent: Any, record: ToolFailureRecord) -> None:
    """Append to the agent's per-execution failure list when it has one."""
    failures = getattr(agent, "_tool_failures", None)
    if isinstance(failures, list):
        failures.append(record)


def handle_tool_failure(
    failure: ToolFailure,
    *,
    tool_name: str,
    tool_args: dict[str, Any] | str | None = None,
    tool: Any = None,
    agent: BaseAgent | LiteAgent | None = None,
    task: Task | None = None,
    crew: Crew | None = None,
) -> ToolFailureRecord | None:
    """Apply the effective policy to a failure a tool just reported.

    Records it on the agent and emits :class:`ToolFailureDetectedEvent`.
    Returns the record, or ``None`` under :attr:`ToolFailurePolicy.IGNORE`.

    Raises:
        ToolExecutionFailedError: Under :attr:`ToolFailurePolicy.RAISE`.
    """
    policy = resolve_tool_failure_policy(tool=tool, agent=agent, task=task, crew=crew)
    if policy is ToolFailurePolicy.IGNORE:
        return None

    record = ToolFailureRecord(
        tool_name=tool_name,
        failure=failure,
        tool_args=tool_args,
        agent_role=getattr(agent, "role", None),
        task_name=(task.name or task.description) if task else None,
        task_id=str(task.id) if task else None,
    )

    _record_on_agent(agent, record)

    # Local import: crewai.events imports tool types back, so a module-level
    # import would cycle.
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolFailureDetectedEvent

    crewai_event_bus.emit(
        agent,
        ToolFailureDetectedEvent(
            tool_name=tool_name,
            tool_args=tool_args if tool_args is not None else {},
            failure=failure,
            policy=policy,
            agent_role=record.agent_role,
            agent_key=getattr(agent, "key", None),
            agent=agent,
            task_name=record.task_name,
            task_id=record.task_id,
        ),
    )

    if policy is ToolFailurePolicy.RAISE:
        raise ToolExecutionFailedError(record)

    return record
