from datetime import datetime
from typing import Any, Callable, Dict

from .base_events import CrewEvent


class ToolUsageEvent(CrewEvent):
    """Base event for tool usage tracking"""

    agent_key: str
    agent_role: str
    tool_name: str
    tool_args: Dict[str, Any] | str
    tool_class: str
    run_attempts: int | None = None
    delegations: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class ToolUsageStartedEvent(ToolUsageEvent):
    """Event emitted when a tool execution is started"""

    type: str = "tool_usage_started"


class ToolUsageFinishedEvent(ToolUsageEvent):
    """Event emitted when a tool execution is completed"""

    started_at: datetime
    finished_at: datetime
    from_cache: bool = False
    type: str = "tool_usage_finished"


class ToolUsageErrorEvent(ToolUsageEvent):
    """Event emitted when a tool execution encounters an error"""

    error: Any
    type: str = "tool_usage_error"


class ToolValidateInputErrorEvent(ToolUsageEvent):
    """Event emitted when a tool input validation encounters an error"""

    error: Any
    type: str = "tool_validate_input_error"


class ToolSelectionErrorEvent(ToolUsageEvent):
    """Event emitted when a tool selection encounters an error"""

    error: Any
    type: str = "tool_selection_error"


class ToolExecutionErrorEvent(CrewEvent):
    """Event emitted when a tool execution encounters an error"""

    error: Any
    type: str = "tool_execution_error"
    tool_name: str
    tool_args: Dict[str, Any]
    tool_class: Callable
