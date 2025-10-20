from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, PrivateAttr

from crewai.agents.agent_builder.base_agent import BaseAgent


if TYPE_CHECKING:
    from crewai.tools.base_tool import BaseTool


class BaseAgentAdapter(BaseAgent, ABC):
    """Base class for all agent adapters in CrewAI.

    This abstract class defines the common interface and functionality that all
    agent adapters must implement. It extends BaseAgent to maintain compatibility
    with the CrewAI framework while adding adapter-specific requirements.
    """

    adapted_structured_output: bool = False
    _agent_config: dict[str, Any] | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, agent_config: dict[str, Any] | None = None, **kwargs: Any):
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
