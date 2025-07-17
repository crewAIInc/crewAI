from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    Union,
)

from crewai.agent import Agent
from crewai.task import Task
from crewai.tools.base_tool import BaseTool


class MemoryConfig(TypedDict, total=False):
    """Typed dictionary for memory configuration."""
    user_memory: Dict[str, Any]
    provider: str
    long_term: bool
    short_term: bool


class CrewConfig(TypedDict, total=False):
    """Typed dictionary for crew configuration."""
    tasks: List[Task]
    agents: List[Agent]
    memory: bool
    memory_config: MemoryConfig
    max_rpm: int
    verbose: bool


class AgentProtocol(Protocol):
    """Protocol for defining agent-like objects."""
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    role: str
    goal: str
    backstory: str
    tools: List[BaseTool]


class TaskProtocol(Protocol):
    """Protocol for defining task-like objects."""
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    description: str
    expected_output: str
    agent: AgentProtocol


def validate_agent(agent: Any) -> bool:
    """Validate if an object conforms to the AgentProtocol."""
    required_attrs = ['role', 'goal', 'backstory', 'tools']
    return all(hasattr(agent, attr) for attr in required_attrs)


def validate_task(task: Any) -> bool:
    """Validate if an object conforms to the TaskProtocol."""
    required_attrs = ['description', 'expected_output', 'agent']
    return all(hasattr(task, attr) for attr in required_attrs)
