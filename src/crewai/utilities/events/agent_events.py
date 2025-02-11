from typing import Any, Dict, List

from crewai.tools.base_tool import BaseTool

from .crew_events import CrewEvent


class AgentExecutorCreated(CrewEvent):
    """Event emitted when an agent executor is created"""

    agent: Any
    tools: List[BaseTool]
    type: str = "agent_executor_created"


class AgentExecutionStarted(CrewEvent):
    """Event emitted when an agent starts executing a task"""

    agent: Any  # type: ignore
    task: Any  # type: ignore
    tools: List[Any]
    inputs: Dict[str, Any]
    type: str = "agent_execution_started"

    model_config = {"arbitrary_types_allowed": True}


class AgentExecutionCompleted(CrewEvent):
    """Event emitted when an agent completes executing a task"""

    agent: Any
    task: Any
    output: str
    type: str = "agent_execution_completed"

    model_config = {"arbitrary_types_allowed": True}


class AgentExecutionError(CrewEvent):
    """Event emitted when an agent encounters an error during execution"""

    agent: Any
    task: Any
    error: str
    type: str = "agent_execution_error"

    model_config = {"arbitrary_types_allowed": True}
