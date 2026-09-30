from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import ConfigDict

from crewai.events.base_events import BaseEvent
from crewai.tools.tool_failure import ToolFailure, ToolFailurePolicy


class ToolUsageEvent(BaseEvent):
    """Base event for tool usage tracking"""

    agent_key: str | None = None
    agent_role: str | None = None
    agent_id: str | None = None
    tool_name: str
    tool_args: dict[str, Any] | str
    tool_class: str | None = None
    run_attempts: int = 0
    delegations: int | None = None
    agent: Any | None = None
    task_name: str | None = None
    task_id: str | None = None
    plan_step_number: int | None = None
    plan_step_description: str | None = None
    from_task: Any | None = None
    from_agent: Any | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        if data.get("from_task"):
            task = data["from_task"]
            data["task_id"] = str(task.id)
            data["task_name"] = task.name or task.description
            data["from_task"] = None

        if data.get("from_agent"):
            agent = data["from_agent"]
            data["agent_id"] = str(agent.id)
            data["agent_role"] = agent.role
            data["from_agent"] = None

        super().__init__(**data)

        if self.agent and hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata


class ToolUsageStartedEvent(ToolUsageEvent):
    """Event emitted when a tool execution is started"""

    type: Literal["tool_usage_started"] = "tool_usage_started"


class ToolUsageFinishedEvent(ToolUsageEvent):
    """Event emitted when a tool execution is completed"""

    started_at: datetime
    finished_at: datetime
    from_cache: bool = False
    output: Any
    failure: ToolFailure | None = None
    """Set when the tool ran but reported it did not succeed.

    Lets a trace UI mark the call failed without correlating a second event.
    """
    type: Literal["tool_usage_finished"] = "tool_usage_finished"


class ToolUsageErrorEvent(ToolUsageEvent):
    """Event emitted when a tool execution encounters an error"""

    error: Any
    type: Literal["tool_usage_error"] = "tool_usage_error"


class ToolFailureDetectedEvent(ToolUsageEvent):
    """Event emitted when a tool completed but reported that it failed.

    Distinct from :class:`ToolUsageErrorEvent`, which covers a tool *raising*.
    This is the quieter case: the call returned normally and says the work was
    not done. Emitted for every policy except ``IGNORE``, and before a
    ``RAISE`` aborts, so subscribers see it even on an aborting run.
    """

    failure: ToolFailure
    policy: ToolFailurePolicy
    type: Literal["tool_failure_detected"] = "tool_failure_detected"


class ToolValidateInputErrorEvent(ToolUsageEvent):
    """Event emitted when a tool input validation encounters an error"""

    error: Any
    type: Literal["tool_validate_input_error"] = "tool_validate_input_error"


class ToolSelectionErrorEvent(ToolUsageEvent):
    """Event emitted when a tool selection encounters an error"""

    error: Any
    type: Literal["tool_selection_error"] = "tool_selection_error"


class ToolExecutionErrorEvent(BaseEvent):
    """Event emitted when a tool execution encounters an error"""

    error: Any
    type: Literal["tool_execution_error"] = "tool_execution_error"
    tool_name: str
    tool_args: dict[str, Any]
    tool_class: Callable[..., Any]
    agent: Any | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.agent and hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata
