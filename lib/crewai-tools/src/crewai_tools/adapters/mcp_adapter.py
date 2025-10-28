"""MCPServer for CrewAI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool

from crewai_tools.adapters.tool_collection import ToolCollection


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mcp import StdioServerParameters
    from mcpadapt.core import MCPAdapt
    from mcpadapt.crewai_adapter import CrewAIAdapter


try:
    from mcp import StdioServerParameters
    from mcpadapt.core import MCPAdapt
    from mcpadapt.crewai_adapter import CrewAIAdapter

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class MCPServerAdapter:
    """Manages the lifecycle of an MCP server and make its tools available to CrewAI.

    Note: tools can only be accessed after the server has been started with the
        `start()` method.

    Attributes:
        tools: The CrewAI tools available from the MCP server.

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
        self._tool_names = list(tool_names) if tool_names else None

        if not MCP_AVAILABLE:
            import click

            if click.confirm(
                "You are missing the 'mcp' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "mcp crewai-tools[mcp]"], check=True)  # noqa: S607

                except subprocess.CalledProcessError as e:
                    raise ImportError("Failed to install mcp package") from e
            else:
                raise ImportError(
                    "`mcp` package not found, please run `uv add crewai-tools[mcp]`"
                )

        try:
            self._serverparams = serverparams
            self._adapter = MCPAdapt(
                self._serverparams, CrewAIAdapter(), connect_timeout
            )
            self.start()

        except Exception as e:
            if self._adapter is not None:
                try:
                    self.stop()
                except Exception as stop_e:
                    logger.error(f"Error during stop cleanup: {stop_e}")
            raise RuntimeError(f"Failed to initialize MCP Adapter: {e}") from e

    def start(self):
        """Start the MCP server and initialize the tools."""
        self._tools = self._adapter.__enter__()

    def stop(self):
        """Stop the MCP server."""
        self._adapter.__exit__(None, None, None)

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

    def __enter__(self):
        """Enter the context manager. Note that `__init__()` already starts the MCP server.
        So tools should already be available.
        """
        return self.tools

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        return self._adapter.__exit__(exc_type, exc_value, traceback)
