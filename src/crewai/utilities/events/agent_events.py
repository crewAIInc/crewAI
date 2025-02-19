from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Union

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool

from .base_events import CrewEvent

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent


class AgentExecutionStartedEvent(CrewEvent):
    """Event emitted when an agent starts executing a task"""

    agent: BaseAgent
    task: Any
    tools: Optional[Sequence[Union[BaseTool, CrewStructuredTool]]]
    task_prompt: str
    type: str = "agent_execution_started"

    model_config = {"arbitrary_types_allowed": True}


class AgentExecutionCompletedEvent(CrewEvent):
    """Event emitted when an agent completes executing a task"""

    agent: BaseAgent
    task: Any
    output: str
    type: str = "agent_execution_completed"


class AgentExecutionErrorEvent(CrewEvent):
    """Event emitted when an agent encounters an error during execution"""

    agent: BaseAgent
    task: Any
    error: str
    type: str = "agent_execution_error"
