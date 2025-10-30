"""Agent-related events moved to break circular dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import ConfigDict, model_validator

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
    type: str = "agent_execution_started"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def set_fingerprint_data(self):
        """Set fingerprint data from the agent if available."""
        if hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata
        return self


class AgentExecutionCompletedEvent(BaseEvent):
    """Event emitted when an agent completes executing a task"""

    agent: BaseAgent
    task: Any
    output: str
    type: str = "agent_execution_completed"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def set_fingerprint_data(self):
        """Set fingerprint data from the agent if available."""
        if hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata
        return self


class AgentExecutionErrorEvent(BaseEvent):
    """Event emitted when an agent encounters an error during execution"""

    agent: BaseAgent
    task: Any
    error: str
    type: str = "agent_execution_error"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def set_fingerprint_data(self):
        """Set fingerprint data from the agent if available."""
        if hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata
        return self


# New event classes for LiteAgent
class LiteAgentExecutionStartedEvent(BaseEvent):
    """Event emitted when a LiteAgent starts executing"""

    agent_info: dict[str, Any]
    tools: Sequence[BaseTool | CrewStructuredTool] | None
    messages: str | list[dict[str, str]]
    type: str = "lite_agent_execution_started"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LiteAgentExecutionCompletedEvent(BaseEvent):
    """Event emitted when a LiteAgent completes execution"""

    agent_info: dict[str, Any]
    output: str
    type: str = "lite_agent_execution_completed"


class LiteAgentExecutionErrorEvent(BaseEvent):
    """Event emitted when a LiteAgent encounters an error during execution"""

    agent_info: dict[str, Any]
    error: str
    type: str = "lite_agent_execution_error"


# Agent Eval events
class AgentEvaluationStartedEvent(BaseEvent):
    agent_id: str
    agent_role: str
    task_id: str | None = None
    iteration: int
    type: str = "agent_evaluation_started"


class AgentEvaluationCompletedEvent(BaseEvent):
    agent_id: str
    agent_role: str
    task_id: str | None = None
    iteration: int
    metric_category: Any
    score: Any
    type: str = "agent_evaluation_completed"


class AgentEvaluationFailedEvent(BaseEvent):
    agent_id: str
    agent_role: str
    task_id: str | None = None
    iteration: int
    error: str
    type: str = "agent_evaluation_failed"
