"""Manager class for environment tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.experimental.environment_tools.file_read_tool import FileReadTool
from crewai.experimental.environment_tools.file_search_tool import FileSearchTool
from crewai.experimental.environment_tools.grep_tool import GrepTool
from crewai.experimental.environment_tools.list_dir_tool import ListDirTool


if TYPE_CHECKING:
    from crewai.tools.base_tool import BaseTool


class EnvironmentTools:
    """Manager class for file system/environment tools.

    Provides a convenient way to create a set of file system tools
    with shared security configuration (allowed_paths).

    Similar to AgentTools but for file system operations. Use this to
    give agents the ability to explore and read files for context engineering.

    Example:
        from crewai.experimental import EnvironmentTools

        # Create tools with security sandbox
        env_tools = EnvironmentTools(
            allowed_paths=["./src", "./docs"],
        )

        # Use with an agent
        agent = Agent(
            role="Code Analyst",
            tools=env_tools.tools(),
        )
    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        include_grep: bool = True,
        include_search: bool = True,
    ) -> None:
        """Initialize EnvironmentTools.

        Args:
            allowed_paths: List of paths to restrict operations to.
                Empty or None allows access to all paths.
            include_grep: Whether to include GrepTool (requires grep installed).
            include_search: Whether to include FileSearchTool.
        """
        self.allowed_paths = allowed_paths or []
        self.include_grep = include_grep
        self.include_search = include_search

    def tools(self) -> list[BaseTool]:
        """Get all configured environment tools.

        Returns:
            List of BaseTool instances with shared allowed_paths configuration.
        """
        tool_list: list[BaseTool] = [
            FileReadTool(allowed_paths=self.allowed_paths),
            ListDirTool(allowed_paths=self.allowed_paths),
        ]

        if self.include_grep:
            tool_list.append(GrepTool(allowed_paths=self.allowed_paths))

        if self.include_search:
            tool_list.append(FileSearchTool(allowed_paths=self.allowed_paths))

        return tool_list
