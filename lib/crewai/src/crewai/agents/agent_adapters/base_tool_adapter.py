from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from crewai.utilities.string_utils import sanitize_tool_name as _sanitize_tool_name


if TYPE_CHECKING:
    from crewai.tools.base_tool import BaseTool


class BaseToolAdapter(ABC):
    """Base class for all tool adapters in CrewAI.

    This abstract class defines the common interface that all tool adapters
    must implement. It provides the structure for adapting CrewAI tools to
    different frameworks and platforms.
    """

    def __init__(self, tools: list[BaseTool] | None = None):
        self.original_tools = tools or []
        self.converted_tools: list[Any] = []

    @abstractmethod
    def configure_tools(self, tools: list[BaseTool]) -> None:
        """Configure and convert tools for the specific implementation.

        Args:
            tools: List of BaseTool instances to be configured and converted
        """

    def tools(self) -> list[Any]:
        """Return all converted tools."""
        return self.converted_tools

    @staticmethod
    def sanitize_tool_name(tool_name: str) -> str:
        """Sanitize tool name for API compatibility."""
        return _sanitize_tool_name(tool_name)
