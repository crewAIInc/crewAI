from abc import ABC, abstractmethod
from typing import Any, List, Optional

from crewai.tools.base_tool import BaseTool


class BaseToolAdapter(ABC):
    """Base class for all tool adapters in CrewAI.

    This abstract class defines the common interface that all tool adapters
    must implement. It provides the structure for adapting CrewAI tools to
    different frameworks and platforms.
    """

    original_tools: List[BaseTool]
    converted_tools: List[Any]

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.original_tools = tools or []
        self.converted_tools = []

    @abstractmethod
    def configure_tools(self, tools: List[BaseTool]) -> None:
        """Configure and convert tools for the specific implementation.

        Args:
            tools: List of BaseTool instances to be configured and converted
        """
        pass

    def tools(self) -> List[Any]:
        """Return all converted tools."""
        return self.converted_tools

    def sanitize_tool_name(self, tool_name: str) -> str:
        """Sanitize tool name for API compatibility."""
        return tool_name.replace(" ", "_")
