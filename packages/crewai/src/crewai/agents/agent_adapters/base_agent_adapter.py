from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import PrivateAttr

from crewai.agent import BaseAgent
from crewai.tools import BaseTool


class BaseAgentAdapter(BaseAgent, ABC):
    """Base class for all agent adapters in CrewAI.

    This abstract class defines the common interface and functionality that all
    agent adapters must implement. It extends BaseAgent to maintain compatibility
    with the CrewAI framework while adding adapter-specific requirements.
    """

    adapted_structured_output: bool = False
    _agent_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, agent_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__(adapted_agent=True, **kwargs)
        self._agent_config = agent_config

    @abstractmethod
    def configure_tools(self, tools: Optional[List[BaseTool]] = None) -> None:
        """Configure and adapt tools for the specific agent implementation.

        Args:
            tools: Optional list of BaseTool instances to be configured
        """
        pass

    def configure_structured_output(self, structured_output: Any) -> None:
        """Configure the structured output for the specific agent implementation.

        Args:
            structured_output: The structured output to be configured
        """
        pass
