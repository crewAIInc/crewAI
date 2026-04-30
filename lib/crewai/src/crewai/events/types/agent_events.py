"""Agent-related events moved to break circular dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.events.base_events import BaseEvent
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool


class AgentExecutionStartedEvent(BaseEvent):
    """Event emitted when an agent starts executing a task"""

    agent: BaseAgent
    task: Any
    tools: Sequence[BaseTool | CrewStructuredTool] | None
    task_prompt: str
    type: Literal["agent_execution_started"] = "agent_execution_started"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def set_fingerprint_data(self) -> Self:
        """Set fingerprint data from the agent if available."""
        _set_agent_fingerprint(self, self.agent)
        return self


class AgentExecutionCompletedEvent(BaseEvent):
    """Event emitted when an agent completes executing a task"""

    agent: BaseAgent
    task: Any
    output: str
    type: Literal["agent_execution_completed"] = "agent_execution_completed"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def set_fingerprint_data(self) -> Self:
        """Set fingerprint data from the agent if available."""
        _set_agent_fingerprint(self, self.agent)
        return self


class AgentExecutionErrorEvent(BaseEvent):
    """Event emitted when an agent encounters an error during execution"""

    agent: BaseAgent
    task: Any
    error: str
    type: Literal["agent_execution_error"] = "agent_execution_error"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def set_fingerprint_data(self) -> Self:
        """Set fingerprint data from the agent if available."""
        _set_agent_fingerprint(self, self.agent)
        return self


# New event classes for LiteAgent
class LiteAgentExecutionStartedEvent(BaseEvent):
    """Event emitted when a LiteAgent starts executing"""

    agent_info: dict[str, Any]
    tools: Sequence[BaseTool | CrewStructuredTool] | None
    messages: str | list[dict[str, str]]
    type: Literal["lite_agent_execution_started"] = "lite_agent_execution_started"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LiteAgentExecutionCompletedEvent(BaseEvent):
    """Event emitted when a LiteAgent completes execution"""

    agent_info: dict[str, Any]
    output: str
    type: Literal["lite_agent_execution_completed"] = "lite_agent_execution_completed"


class LiteAgentExecutionErrorEvent(BaseEvent):
    """Event emitted when a LiteAgent encounters an error during execution"""

    agent_info: dict[str, Any]
    error: str
    type: Literal["lite_agent_execution_error"] = "lite_agent_execution_error"


# Agent Eval events
class AgentEvaluationStartedEvent(BaseEvent):
    agent_id: str
    agent_role: str
    task_id: str | None = None
    iteration: int
    type: Literal["agent_evaluation_started"] = "agent_evaluation_started"


class AgentEvaluationCompletedEvent(BaseEvent):
    agent_id: str
    agent_role: str
    task_id: str | None = None
    iteration: int
    metric_category: Any
    score: Any
    type: Literal["agent_evaluation_completed"] = "agent_evaluation_completed"


class AgentEvaluationFailedEvent(BaseEvent):
    agent_id: str
    agent_role: str
    task_id: str | None = None
    iteration: int
    error: str
    type: Literal["agent_evaluation_failed"] = "agent_evaluation_failed"


def _set_agent_fingerprint(event: BaseEvent, agent: BaseAgent) -> None:
    """Set fingerprint data on an event from an agent object."""
    fp = agent.security_config.fingerprint
    if fp is not None:
        event.source_fingerprint = fp.uuid_str
        event.source_type = "agent"
        if fp.metadata:
            event.fingerprint_metadata = fp.metadata
