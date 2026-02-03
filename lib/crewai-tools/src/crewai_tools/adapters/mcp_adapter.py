"""MCPServer for CrewAI."""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool
from crewai.utilities.pydantic_schema_utils import (
    create_model_from_schema,
)
from crewai.utilities.string_utils import sanitize_tool_name
from pydantic import BaseModel

from crewai_tools.adapters.tool_collection import ToolCollection


if TYPE_CHECKING:
    from mcp import StdioServerParameters
    from mcp.types import CallToolResult, TextContent, Tool
    from mcpadapt.core import MCPAdapt, ToolAdapter


logger = logging.getLogger(__name__)


try:
    from mcp import StdioServerParameters
    from mcp.types import CallToolResult, TextContent, Tool
    from mcpadapt.core import MCPAdapt, ToolAdapter

    class CrewAIToolAdapter(ToolAdapter):
        """Adapter that creates CrewAI tools with properly normalized JSON schemas.

        This adapter bypasses mcpadapt's model creation which adds invalid null values
        to field schemas, instead using CrewAI's own schema utilities.
        """

        def adapt(
            self,
            func: Callable[[dict[str, Any] | None], CallToolResult],
            mcp_tool: Tool,
        ) -> BaseTool:
            """Adapt a MCP tool to a CrewAI tool.

            Args:
                func: The function to call when the tool is invoked.
                mcp_tool: The MCP tool definition to adapt.

            Returns:
                A CrewAI BaseTool instance.
            """
            tool_name = sanitize_tool_name(mcp_tool.name)
            tool_description = mcp_tool.description or ""
            input_schema = mcp_tool.inputSchema

            args_model = create_model_from_schema(input_schema)

            class CrewAIMCPTool(BaseTool):
                name: str = tool_name
                description: str = tool_description
                args_schema: type[BaseModel] = args_model

                def _run(self, **kwargs: Any) -> Any:
                    filtered_kwargs: dict[str, Any] = {}
                    schema_properties = input_schema.get("properties", {})

                    for key, value in kwargs.items():
                        if value is None and key in schema_properties:
                            prop_schema = schema_properties[key]
                            if isinstance(prop_schema.get("type"), list):
                                if "null" in prop_schema["type"]:
                                    filtered_kwargs[key] = value
                            elif "anyOf" in prop_schema:
                                if any(
                                    opt.get("type") == "null"
                                    for opt in prop_schema["anyOf"]
                                ):
                                    filtered_kwargs[key] = value
                        else:
                            filtered_kwargs[key] = value

                    result = func(filtered_kwargs)
                    if len(result.content) == 1:
                        first_content = result.content[0]
                        if isinstance(first_content, TextContent):
                            return first_content.text
                        return str(first_content)
                    return str(
                        [
                            content.text
                            for content in result.content
                            if isinstance(content, TextContent)
                        ]
                    )

                def _generate_description(self) -> None:
                    schema = self.args_schema.model_json_schema()
                    schema.pop("$defs", None)
                    self.description = (
                        f"Tool Name: {self.name}\n"
                        f"Tool Arguments: {schema}\n"
                        f"Tool Description: {self.description}"
                    )

            return CrewAIMCPTool()

        async def async_adapt(self, afunc: Any, mcp_tool: Tool) -> Any:
            """Async adaptation is not supported by CrewAI."""
            raise NotImplementedError("async is not supported by the CrewAI framework.")

    MCP_AVAILABLE = True
except ImportError as e:
    logger.debug(f"MCP packages not available: {e}")
    MCP_AVAILABLE = False


class MCPServerAdapter:
    """Manages the lifecycle of an MCP server and make its tools available to CrewAI.

    Note: tools can only be accessed after the server has been started with the
        `start()` method.

    Usage:
        # context manager + stdio
        with MCPServerAdapter(...) as tools:
            # tools is now available

        # context manager + sse
        with MCPServerAdapter({"url": "http://localhost:8000/sse"}) as tools:
            # tools is now available

        # context manager with filtered tools
        with MCPServerAdapter(..., "tool1", "tool2") as filtered_tools:
            # only tool1 and tool2 are available

        # context manager with custom connect timeout (60 seconds)
        with MCPServerAdapter(..., connect_timeout=60) as tools:
            # tools is now available with longer timeout

        # manually stop mcp server
        try:
            mcp_server = MCPServerAdapter(...)
            tools = mcp_server.tools  # all tools

            # or with filtered tools and custom timeout
            mcp_server = MCPServerAdapter(..., "tool1", "tool2", connect_timeout=45)
            filtered_tools = mcp_server.tools  # only tool1 and tool2
            ...
        finally:
            mcp_server.stop()

        # Best practice is ensure cleanup is done after use.
        mcp_server.stop() # run after crew().kickoff()
    """

    def __init__(
        self,
        serverparams: StdioServerParameters | dict[str, Any],
        *tool_names: str,
        connect_timeout: int = 30,
    ) -> None:
        """Initialize the MCP Server.

        Args:
            serverparams: The parameters for the MCP server it supports either a
                `StdioServerParameters` or a `dict` respectively for STDIO and SSE.
            *tool_names: Optional names of tools to filter. If provided, only tools with
                matching names will be available.
            connect_timeout: Connection timeout in seconds to the MCP server (default is 30s).

        """
        super().__init__()
        self._adapter = None
        self._tools = None
        self._tool_names = (
            [sanitize_tool_name(name) for name in tool_names] if tool_names else None
        )

        if not MCP_AVAILABLE:
            import click

            if click.confirm(
                "You are missing the 'mcp' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "mcp crewai-tools'[mcp]'"], check=True)  # noqa: S607

                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install mcp package") from e
            else:
                raise ImportError(
                    "`mcp` package not found, please run `uv add crewai-tools[mcp]`"
                )

        try:
            self._serverparams = serverparams
            self._adapter = MCPAdapt(
                self._serverparams, CrewAIToolAdapter(), connect_timeout
            )
            self.start()

        except Exception as e:
            if self._adapter is not None:
                try:
                    self.stop()
                except Exception as stop_e:
                    logger.error(f"Error during stop cleanup: {stop_e}")
            raise RuntimeError(f"Failed to initialize MCP Adapter: {e}") from e

    def start(self) -> None:
        """Start the MCP server and initialize the tools."""
        self._tools = self._adapter.__enter__()  # type: ignore[union-attr]

    def stop(self) -> None:
        """Stop the MCP server."""
        self._adapter.__exit__(None, None, None)  # type: ignore[union-attr]

    @property
    def tools(self) -> ToolCollection[BaseTool]:
        """The CrewAI tools available from the MCP server.

        Raises:
            ValueError: If the MCP server is not started.

        Returns:
            The CrewAI tools available from the MCP server.
        """
        if self._tools is None:
            raise ValueError(
                "MCP server not started, run `mcp_server.start()` first before accessing `tools`"
            )

        tools_collection = ToolCollection(self._tools)
        if self._tool_names:
            return tools_collection.filter_by_names(self._tool_names)
        return tools_collection

    def __enter__(self) -> ToolCollection[BaseTool]:
        """Enter the context manager.

        Note that `__init__()` already starts the MCP server,
        so tools should already be available.
        """
        return self.tools

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Exit the context manager."""
        self._adapter.__exit__(exc_type, exc_value, traceback)  # type: ignore[union-attr]
