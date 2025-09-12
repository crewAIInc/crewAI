"""Base adapter for integrating external agent implementations with CrewAI."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypedDict

from pydantic import ConfigDict, PrivateAttr
from typing_extensions import Unpack

from crewai.agent import BaseAgent
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.security.security_config import SecurityConfig
from crewai.tools import BaseTool
from crewai.utilities import I18N


class AgentKwargs(TypedDict, total=False):
    """TypedDict for BaseAgent initialization arguments."""

    role: str
    goal: str
    backstory: str
    config: dict[str, Any] | None
    cache: bool
    verbose: bool
    max_rpm: int | None
    allow_delegation: bool
    tools: list[BaseTool] | None
    max_iter: int
    llm: Any
    crew: Any
    i18n: I18N
    max_tokens: int | None
    knowledge: Knowledge | None
    knowledge_sources: list[BaseKnowledgeSource] | None
    knowledge_storage: Any
    security_config: SecurityConfig
    callbacks: list[Callable[..., Any]]


class BaseAgentAdapter(BaseAgent, ABC):
    """Base class for all agent adapters in CrewAI.

    This abstract class defines the common interface and functionality that all
    agent adapters must implement. It extends BaseAgent to maintain compatibility
    with the CrewAI framework while adding adapter-specific requirements.
    """

    adapted_structured_output: bool = False
    _agent_config: dict[str, Any] | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        agent_config: dict[str, Any] | None = None,
        **kwargs: Unpack[AgentKwargs],
    ) -> None:
        """Initialize the base agent adapter.

        Args:
            agent_config: Optional configuration dictionary for the adapted agent.
            **kwargs: BaseAgent initialization arguments (role, goal, backstory, etc).
        """
        super().__init__(adapted_agent=True, **kwargs)
        self._agent_config = agent_config

    @abstractmethod
    def configure_tools(self, tools: list[BaseTool] | None = None) -> None:
        """Configure and adapt tools for the specific agent implementation.

        Args:
            tools: Optional list of BaseTool instances to be configured
        """

    def configure_structured_output(self, structured_output: Any) -> None:
        """Configure the structured output for the specific agent implementation.

        Args:
            structured_output: The structured output to be configured
        """
