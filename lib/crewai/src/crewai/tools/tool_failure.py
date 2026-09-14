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

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
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


def merge_tool_failures(
    *groups: list[ToolFailureRecord],
) -> list[ToolFailureRecord]:
    """Concatenate failure lists, dropping records already present.

    Guardrail retries rebuild the output from overlapping sources, so identity
    is not enough to avoid duplicates.
    """
    merged: list[ToolFailureRecord] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for record in group:
            key = (
                record.tool_name,
                record.failure.message,
                record.failure.code,
                record.task_id,
                str(record.tool_args),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)
    return merged


def collect_tool_failures(agent: Any) -> list[ToolFailureRecord]:
    """Failures for the execution in progress, else the agent's last ones.

    Prefers the active collector so a shared agent running concurrent tasks
    reports only the caller's own records. Tolerates agents that do not expose
    the attribute at all, since reading telemetry must never raise.
    """
    active = active_tool_failures()
    if active is not None:
        return list(active)

    records = getattr(agent, "_tool_failures", None)
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, ToolFailureRecord)]


def _agent_id(agent: Any) -> str | None:
    """Stringified agent id, for correlating events with the call."""
    agent_id = getattr(agent, "id", None)
    return str(agent_id) if agent_id is not None else None


_active_failures: ContextVar[list[ToolFailureRecord] | None] = ContextVar(
    "crewai_tool_failures", default=None
)


@contextmanager
def tool_failure_collector() -> Generator[list[ToolFailureRecord], None, None]:
    """Collect the failures of one execution, isolated from concurrent ones.

    An agent may be shared by tasks running concurrently, so accumulating on
    the agent lets one execution erase or inherit another's records. The
    collector is a ContextVar, which asyncio tasks and threads copy, so each
    execution sees only its own. Nesting is safe: a guardrail retry can open
    its own scope inside the outer one.
    """
    records: list[ToolFailureRecord] = []
    token = _active_failures.set(records)
    try:
        yield records
    finally:
        _active_failures.reset(token)


def active_tool_failures() -> list[ToolFailureRecord] | None:
    """Records for the execution in progress, or None outside a collector."""
    return _active_failures.get()


def _record_failure(agent: Any, record: ToolFailureRecord) -> None:
    """Store a record on the active collector and on the agent.

    The collector is what outputs read, so it is authoritative. The agent copy
    only backs ``last_tool_failures``, which reports the most recent execution
    in the same best-effort way ``last_messages`` does.
    """
    records = _active_failures.get()
    if records is not None:
        records.append(record)

    failures = getattr(agent, "_tool_failures", None)
    if isinstance(failures, list):
        failures.append(record)


def reportable_failure(
    failure: ToolFailure | None,
    *,
    tool: Any = None,
    agent: BaseAgent | LiteAgent | None = None,
    task: Task | None = None,
    crew: Crew | None = None,
) -> ToolFailure | None:
    """Return the failure to attach to ``ToolUsageFinishedEvent``.

    ``None`` under :attr:`ToolFailurePolicy.IGNORE`, so that policy really does
    surface nothing -- neither a record, nor an event, nor a flag on the
    finished event that a trace UI would render as a failure.
    """
    if failure is None:
        return None
    policy = resolve_tool_failure_policy(tool=tool, agent=agent, task=task, crew=crew)
    return None if policy is ToolFailurePolicy.IGNORE else failure


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

    _record_failure(agent, record)

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
            # Set explicitly rather than via from_agent, which would also
            # overwrite agent_role and lose the _original_role preference that
            # the paired ToolUsage events use.
            agent_id=_agent_id(agent),
            agent=agent,
            task_name=record.task_name,
            task_id=record.task_id,
        ),
    )

    if policy is ToolFailurePolicy.RAISE:
        raise ToolExecutionFailedError(record)

    return record
