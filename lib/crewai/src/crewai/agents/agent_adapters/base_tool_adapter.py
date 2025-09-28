from abc import ABC, abstractmethod
from typing import Any

from crewai.tools.base_tool import BaseTool


class BaseToolAdapter(ABC):
    """Base class for all tool adapters in CrewAI.

    This abstract class defines the common interface that all tool adapters
    must implement. It provides the structure for adapting CrewAI tools to
    different frameworks and platforms.
    """

    original_tools: list[BaseTool]
    converted_tools: list[Any]

    def __init__(self, tools: list[BaseTool] | None = None):
        self.original_tools = tools or []
        self.converted_tools = []

    @abstractmethod
    def configure_tools(self, tools: list[BaseTool]) -> None:
        """Configure and convert tools for the specific implementation.

        Args:
            tools: List of BaseTool instances to be configured and converted
        """

    def tools(self) -> list[Any]:
        """Return all converted tools."""
        return self.converted_tools

    def sanitize_tool_name(self, tool_name: str) -> str:
        """Sanitize tool name for API compatibility."""
        return tool_name.replace(" ", "_")
