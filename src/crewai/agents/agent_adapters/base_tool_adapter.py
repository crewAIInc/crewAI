"""Base adapter for tool conversions across different agent frameworks."""

from abc import ABC, abstractmethod
from typing import Any

from crewai.tools.base_tool import BaseTool


class BaseToolAdapter(ABC):
    """Base class for all tool adapters in CrewAI.

    This abstract class defines the common interface that all tool adapters
    must implement. It provides the structure for adapting CrewAI tools to
    different frameworks and platforms.
    """

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        """Initialize the tool adapter.

        Args:
            tools: Optional list of BaseTool instances to be adapted.
        """
        self.original_tools: list[BaseTool] = tools or []
        self.converted_tools: list[Any] = []

    @abstractmethod
    def configure_tools(self, tools: list[BaseTool]) -> None:
        """Configure and convert tools for the specific implementation.

        Args:
            tools: List of BaseTool instances to be configured and converted.
        """

    def tools(self) -> list[Any]:
        """Return all converted tools.

        Returns:
            List of tools converted to the target framework format.
        """
        return self.converted_tools

    @staticmethod
    def sanitize_tool_name(tool_name: str) -> str:
        """Sanitize tool name for API compatibility.

        Args:
            tool_name: Original tool name that may contain spaces or special characters.

        Returns:
            Sanitized tool name with underscores replacing spaces.
        """
        return tool_name.replace(" ", "_")
