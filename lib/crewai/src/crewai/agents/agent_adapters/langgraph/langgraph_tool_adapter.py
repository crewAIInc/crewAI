"""LangGraph tool adapter for CrewAI tool integration.

This module contains the LangGraphToolAdapter class that converts CrewAI tools
to LangGraph-compatible format using langchain_core.tools.
"""

import inspect
from typing import Any

from crewai.agents.agent_adapters.base_tool_adapter import BaseToolAdapter
from crewai.tools.base_tool import BaseTool


class LangGraphToolAdapter(BaseToolAdapter):
    """Adapts CrewAI tools to LangGraph agent tool compatible format.

    Converts CrewAI BaseTool instances to langchain_core.tools format
    that can be used by LangGraph agents.
    """

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        """Initialize the tool adapter.

        Args:
            tools: Optional list of CrewAI tools to adapt.
        """
        super().__init__()
        self.original_tools: list[BaseTool] = tools or []
        self.converted_tools: list[Any] = []

    async def _safe_execute(self, tool: BaseTool, *args: Any, **kwargs: Any) -> Any:
        """Safely execute a tool with error handling.

        Args:
            tool: The CrewAI tool instance to execute.
            *args: Positional arguments to pass to the tool.
            **kwargs: Keyword arguments to pass to the tool.

        Returns:
            The result of the tool execution, or a dictionary containing
            error information if execution fails.
        """
        try:
            output = tool.run(*args, **kwargs)

            if inspect.isawaitable(output):
                return await output

            return output

        except Exception as e:
            return {
                "error": str(e),
                "tool": tool.name,
            }

    def _make_tool_wrapper(self, tool: BaseTool):
        """Create a wrapper function for a CrewAI tool compatible with LangGraph.

        This method generates an async wrapper that standardizes input handling
        and delegates execution to the safe execution layer.

        Args:
            tool: The CrewAI tool instance to wrap.

        Returns:
            An async callable that executes the tool with safe error handling.
        """

        async def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Execute the wrapped tool with standardized input handling.

            Args:
                *args: Positional arguments for the tool.
                **kwargs: Keyword arguments for the tool.

            Returns:
                The tool execution result, or a dictionary with keys
                "error" and "tool" if execution fails.
            """
            if len(args) > 0 and isinstance(args[0], str):
                return await self._safe_execute(tool, args[0])
            elif "input" in kwargs:
                return await self._safe_execute(tool, kwargs["input"])
            else:
                return await self._safe_execute(tool, **kwargs)

        return tool_wrapper

    def configure_tools(self, tools: list[BaseTool]) -> None:
        """Configure and convert CrewAI tools to LangGraph-compatible format.

        LangGraph expects tools in langchain_core.tools format. This method
        converts CrewAI BaseTool instances to StructuredTool instances.

        Args:
            tools: List of CrewAI tools to convert.
        """
        from langchain_core.tools import BaseTool as LangChainBaseTool, StructuredTool

        converted_tools: list[Any] = []
        if self.original_tools:
            all_tools: list[BaseTool] = tools + self.original_tools
        else:
            all_tools = tools
        for tool in all_tools:
            if isinstance(tool, LangChainBaseTool):
                converted_tools.append(tool)
                continue

            sanitized_name: str = self.sanitize_tool_name(tool.name)

            converted_tool: StructuredTool = StructuredTool(
                name=sanitized_name,
                description=tool.description,
                func=self._make_tool_wrapper(tool),
                args_schema=tool.args_schema,
            )

            converted_tools.append(converted_tool)

        self.converted_tools = converted_tools

    def tools(self) -> list[Any]:
        """Get the list of converted tools.

        Returns:
            List of LangGraph-compatible tools.
        """
        return self.converted_tools or []
