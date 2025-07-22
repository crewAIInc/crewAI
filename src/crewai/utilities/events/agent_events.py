from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool

from .base_events import BaseEvent

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent


class AgentExecutionStartedEvent(BaseEvent):
    """Event emitted when an agent starts executing a task"""

    agent: BaseAgent
    task: Any
    tools: Optional[Sequence[Union[BaseTool, CrewStructuredTool]]]
    task_prompt: str
    type: str = "agent_execution_started"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the agent
        if hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata


class AgentExecutionCompletedEvent(BaseEvent):
    """Event emitted when an agent completes executing a task"""

    agent: BaseAgent
    task: Any
    output: str
    type: str = "agent_execution_completed"

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the agent
        if hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata


class AgentExecutionErrorEvent(BaseEvent):
    """Event emitted when an agent encounters an error during execution"""

    agent: BaseAgent
    task: Any
    error: str
    type: str = "agent_execution_error"

    def __init__(self, **data):
        super().__init__(**data)
        # Set fingerprint data from the agent
        if hasattr(self.agent, "fingerprint") and self.agent.fingerprint:
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata


# New event classes for LiteAgent
class LiteAgentExecutionStartedEvent(BaseEvent):
    """Event emitted when a LiteAgent starts executing"""

    agent_info: Dict[str, Any]
    tools: Optional[Sequence[Union[BaseTool, CrewStructuredTool]]]
    messages: Union[str, List[Dict[str, str]]]
    type: str = "lite_agent_execution_started"

    model_config = {"arbitrary_types_allowed": True}


class LiteAgentExecutionCompletedEvent(BaseEvent):
    """Event emitted when a LiteAgent completes execution"""

    agent_info: Dict[str, Any]
    output: str
    type: str = "lite_agent_execution_completed"


class LiteAgentExecutionErrorEvent(BaseEvent):
    """Event emitted when a LiteAgent encounters an error during execution"""

    agent_info: Dict[str, Any]
    error: str
    type: str = "lite_agent_execution_error"


# New logging events
class AgentLogsStartedEvent(BaseEvent):
    """Event emitted when agent logs should be shown at start"""

    agent_role: str
    task_description: Optional[str] = None
    verbose: bool = False
    type: str = "agent_logs_started"


class AgentLogsExecutionEvent(BaseEvent):
    """Event emitted when agent logs should be shown during execution"""

    agent_role: str
    formatted_answer: Any
    verbose: bool = False
    type: str = "agent_logs_execution"

    model_config = {"arbitrary_types_allowed": True}

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
