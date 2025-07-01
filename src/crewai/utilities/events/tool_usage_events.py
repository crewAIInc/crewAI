from datetime import datetime
from typing import Any, Callable, Dict, Optional

from .base_events import BaseEvent


class ToolUsageEvent(BaseEvent):
    """Base event for tool usage tracking"""

    agent_key: Optional[str] = None
    agent_role: Optional[str] = None
    tool_name: str
    tool_args: Dict[str, Any] | str
    tool_class: Optional[str] = None
    run_attempts: int | None = None
    delegations: int | None = None
    agent: Optional[Any] = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the agent
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

    type: str = "tool_usage_started"


class ToolUsageFinishedEvent(ToolUsageEvent):
    """Event emitted when a tool execution is completed"""

    started_at: datetime
    finished_at: datetime
    from_cache: bool = False
    output: Any
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


class ToolExecutionErrorEvent(BaseEvent):
    """Event emitted when a tool execution encounters an error"""

    error: Any
    type: str = "tool_execution_error"
    tool_name: str
    tool_args: Dict[str, Any]
    tool_class: Callable
    agent: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the agent
        if self.agent and hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata
