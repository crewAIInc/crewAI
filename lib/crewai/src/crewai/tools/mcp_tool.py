"""MCPTool - User-friendly interface for MCP server tool integration.

This module provides a simple API for connecting to MCP servers and
getting tools that can be used with CrewAI agents.

Example:
    ```python
    from crewai.tools import MCPTool

    # Connect to MCP servers by name (runs via npx)
    cloud_tools = MCPTool.from_server("@anthropic/mcp-server-filesystem")

    # Connect to MCP servers by URL
    api_tools = MCPTool.from_server("https://api.example.com/mcp")

    # Use tools with an agent
    agent = Agent(
        role='Cloud Engineer',
        tools=[*cloud_tools, *api_tools]
    )
    ```
"""

from __future__ import annotations

import asyncio
import shutil
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, create_model

from crewai.tools.base_tool import BaseTool


if TYPE_CHECKING:
    from crewai.mcp import (
        MCPServerConfig,
        MCPServerStdio,
        ToolFilter,
    )


MCP_CONNECTION_TIMEOUT = 30
MCP_TOOL_EXECUTION_TIMEOUT = 30
MCP_DISCOVERY_TIMEOUT = 30


class MCPTool:
    """Factory class for creating tools from MCP servers.

    This class provides a simple interface for connecting to MCP servers
    and retrieving tools that can be used with CrewAI agents.

    Example:
        ```python
        from crewai.tools import MCPTool

        # Get tools from an MCP server by name
        tools = MCPTool.from_server("@anthropic/mcp-server-filesystem")

        # Get tools from an MCP server by URL
        tools = MCPTool.from_server("https://api.example.com/mcp")

        # Get tools with custom configuration
        tools = MCPTool.from_server(
            MCPServerStdio(
                command="npx",
                args=["-y", "@anthropic/mcp-server-filesystem"],
                env={"HOME": "/home/user"}
            )
        )
        ```
    """

    @classmethod
    def from_server(
        cls,
        server: str | MCPServerConfig,
        *,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        tool_filter: ToolFilter | None = None,
        cache_tools_list: bool = False,
    ) -> list[BaseTool]:
        """Get tools from an MCP server.

        This method connects to an MCP server, discovers available tools,
        and returns them as CrewAI BaseTool instances.

        Args:
            server: MCP server specification. Can be:
                - A server name/package (e.g., "@anthropic/mcp-server-filesystem")
                  which will be run via npx or uvx
                - A URL (e.g., "https://api.example.com/mcp") for HTTP/SSE servers
                - An MCPServerConfig object (MCPServerStdio, MCPServerHTTP, MCPServerSSE)
            env: Environment variables for stdio servers (ignored for HTTP/SSE).
            headers: HTTP headers for HTTP/SSE servers (ignored for stdio).
            tool_filter: Optional filter for available tools.
            cache_tools_list: Whether to cache the tool list for faster subsequent access.

        Returns:
            List of BaseTool instances from the MCP server.

        Raises:
            ValueError: If the server specification is invalid.
            ConnectionError: If connection to the MCP server fails.
            ImportError: If the MCP library is not installed.

        Example:
            ```python
            # By server name (runs via npx)
            tools = MCPTool.from_server("@anthropic/mcp-server-filesystem")

            # By URL
            tools = MCPTool.from_server("https://api.example.com/mcp")

            # With environment variables
            tools = MCPTool.from_server(
                "@anthropic/mcp-server-filesystem",
                env={"HOME": "/home/user"}
            )

            # With HTTP headers
            tools = MCPTool.from_server(
                "https://api.example.com/mcp",
                headers={"Authorization": "Bearer token"}
            )

            # With tool filter
            tools = MCPTool.from_server(
                "@anthropic/mcp-server-filesystem",
                tool_filter=lambda tools: [t for t in tools if "read" in t["name"]]
            )
            ```
        """
        config = cls._resolve_server_config(
            server,
            env=env,
            headers=headers,
            tool_filter=tool_filter,
            cache_tools_list=cache_tools_list,
        )
        return cls._get_tools_from_config(config)

    @classmethod
    def _resolve_server_config(
        cls,
        server: str | MCPServerConfig,
        *,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        tool_filter: ToolFilter | None = None,
        cache_tools_list: bool = False,
    ) -> MCPServerConfig:
        """Resolve server specification to MCPServerConfig.

        Args:
            server: Server name, URL, or MCPServerConfig.
            env: Environment variables for stdio servers.
            headers: HTTP headers for HTTP/SSE servers.
            tool_filter: Optional tool filter.
            cache_tools_list: Whether to cache tools list.

        Returns:
            MCPServerConfig instance.
        """
        from crewai.mcp import MCPServerHTTP, MCPServerSSE, MCPServerStdio

        if isinstance(server, (MCPServerStdio, MCPServerHTTP, MCPServerSSE)):
            return server

        if not isinstance(server, str):
            raise ValueError(
                f"Invalid server type: {type(server)}. "
                "Must be a string (server name or URL) or MCPServerConfig."
            )

        if server.startswith(("http://", "https://")):
            return MCPServerHTTP(
                url=server,
                headers=headers,
                tool_filter=tool_filter,
                cache_tools_list=cache_tools_list,
            )

        return cls._create_stdio_config(
            server,
            env=env,
            tool_filter=tool_filter,
            cache_tools_list=cache_tools_list,
        )

    @classmethod
    def _create_stdio_config(
        cls,
        server_name: str,
        *,
        env: dict[str, str] | None = None,
        tool_filter: ToolFilter | None = None,
        cache_tools_list: bool = False,
    ) -> MCPServerStdio:
        """Create stdio config for a server name.

        Determines whether to use npx or uvx based on the server name
        and available executables.

        Args:
            server_name: MCP server package name.
            env: Environment variables.
            tool_filter: Optional tool filter.
            cache_tools_list: Whether to cache tools list.

        Returns:
            MCPServerStdio configuration.
        """
        from crewai.mcp import MCPServerStdio

        if server_name.startswith(("@", "mcp-server-")):
            command, args = cls._get_npx_command(server_name)
        elif server_name.endswith("-mcp-server") or "-mcp" in server_name:
            command, args = cls._get_uvx_command(server_name)
        else:
            command, args = cls._detect_runner(server_name)

        return MCPServerStdio(
            command=command,
            args=args,
            env=env,
            tool_filter=tool_filter,
            cache_tools_list=cache_tools_list,
        )

    @classmethod
    def _get_npx_command(cls, server_name: str) -> tuple[str, list[str]]:
        """Get npx command for running an npm package.

        Args:
            server_name: NPM package name.

        Returns:
            Tuple of (command, args).
        """
        npx_path = shutil.which("npx")
        if npx_path is None:
            raise ValueError(
                f"npx not found. Please install Node.js to use MCP server: {server_name}"
            )
        return "npx", ["-y", server_name]

    @classmethod
    def _get_uvx_command(cls, server_name: str) -> tuple[str, list[str]]:
        """Get uvx command for running a Python package.

        Args:
            server_name: Python package name.

        Returns:
            Tuple of (command, args).
        """
        uvx_path = shutil.which("uvx")
        if uvx_path is None:
            raise ValueError(
                f"uvx not found. Please install uv to use MCP server: {server_name}"
            )
        return "uvx", [server_name]

    @classmethod
    def _detect_runner(cls, server_name: str) -> tuple[str, list[str]]:
        """Detect the appropriate runner for a server name.

        Tries npx first, then uvx.

        Args:
            server_name: Server package name.

        Returns:
            Tuple of (command, args).
        """
        npx_path = shutil.which("npx")
        if npx_path is not None:
            return "npx", ["-y", server_name]

        uvx_path = shutil.which("uvx")
        if uvx_path is not None:
            return "uvx", [server_name]

        raise ValueError(
            f"Neither npx nor uvx found. Please install Node.js or uv to use MCP server: {server_name}"
        )

    @classmethod
    def _get_tools_from_config(cls, config: MCPServerConfig) -> list[BaseTool]:
        """Get tools from an MCP server configuration.

        Args:
            config: MCP server configuration.

        Returns:
            List of BaseTool instances.
        """
        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, cls._get_tools_from_config_async(config)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(cls._get_tools_from_config_async(config))

    @classmethod
    async def _get_tools_from_config_async(
        cls, config: MCPServerConfig
    ) -> list[BaseTool]:
        """Async implementation of getting tools from config.

        Args:
            config: MCP server configuration.

        Returns:
            List of BaseTool instances.
        """
        from crewai.mcp import MCPClient

        transport = cls._create_transport(config)
        client = MCPClient(
            transport,
            connect_timeout=MCP_CONNECTION_TIMEOUT,
            execution_timeout=MCP_TOOL_EXECUTION_TIMEOUT,
            discovery_timeout=MCP_DISCOVERY_TIMEOUT,
            cache_tools_list=config.cache_tools_list,
        )

        try:
            await client.connect()
            tool_schemas = await client.list_tools()

            if config.tool_filter is not None:
                tool_schemas = config.tool_filter(tool_schemas)

            server_name = cls._extract_server_name(config)
            tools = []

            for schema in tool_schemas:
                tool = cls._create_tool_from_schema(
                    schema, config, server_name, client.transport
                )
                tools.append(tool)

            return tools
        finally:
            await client.disconnect()

    @classmethod
    def _create_transport(cls, config: MCPServerConfig) -> Any:
        """Create transport from config.

        Args:
            config: MCP server configuration.

        Returns:
            Transport instance.
        """
        from crewai.mcp import MCPServerHTTP, MCPServerSSE, MCPServerStdio
        from crewai.mcp.transports.http import HTTPTransport
        from crewai.mcp.transports.sse import SSETransport
        from crewai.mcp.transports.stdio import StdioTransport

        if isinstance(config, MCPServerStdio):
            return StdioTransport(
                command=config.command,
                args=config.args,
                env=config.env,
            )
        if isinstance(config, MCPServerHTTP):
            return HTTPTransport(
                url=config.url,
                headers=config.headers,
            )
        if isinstance(config, MCPServerSSE):
            return SSETransport(
                url=config.url,
                headers=config.headers,
            )
        raise ValueError(f"Unsupported MCP server config type: {type(config)}")

    @classmethod
    def _extract_server_name(cls, config: MCPServerConfig) -> str:
        """Extract a human-readable server name from config.

        Args:
            config: MCP server configuration.

        Returns:
            Server name string.
        """
        from crewai.mcp import MCPServerHTTP, MCPServerSSE, MCPServerStdio

        if isinstance(config, MCPServerStdio):
            if config.args:
                last_arg = config.args[-1]
                if last_arg.startswith("@"):
                    parts = last_arg.split("/")
                    return parts[-1] if len(parts) > 1 else last_arg.replace("@", "")
                return last_arg.replace("-", "_").replace(".", "_")
            return config.command
        if isinstance(config, (MCPServerHTTP, MCPServerSSE)):
            from urllib.parse import urlparse

            parsed = urlparse(config.url)
            return parsed.netloc.replace(".", "_").replace("-", "_").replace(":", "_")
        return "mcp_server"

    @classmethod
    def _create_tool_from_schema(
        cls,
        schema: dict[str, Any],
        config: MCPServerConfig,
        server_name: str,
        transport: Any,
    ) -> BaseTool:
        """Create a BaseTool from an MCP tool schema.

        Args:
            schema: Tool schema from MCP server.
            config: MCP server configuration.
            server_name: Server name for prefixing.
            transport: Transport instance for tool execution.

        Returns:
            BaseTool instance.
        """
        tool_name = schema.get("name", "unknown_tool")
        description = schema.get("description", f"Tool {tool_name} from {server_name}")
        input_schema = schema.get("inputSchema", {})

        prefixed_name = f"{server_name}_{tool_name}"

        args_schema = cls._json_schema_to_pydantic(tool_name, input_schema)

        return _MCPToolInstance(
            name=prefixed_name,
            description=description,
            args_schema=args_schema,
            config=config,
            original_tool_name=tool_name,
            server_name=server_name,
        )

    @classmethod
    def _json_schema_to_pydantic(
        cls, tool_name: str, json_schema: dict[str, Any]
    ) -> type[BaseModel]:
        """Convert JSON schema to Pydantic model.

        Args:
            tool_name: Tool name for the model.
            json_schema: JSON schema from MCP server.

        Returns:
            Pydantic model class.
        """
        properties = json_schema.get("properties", {})
        required = set(json_schema.get("required", []))

        fields: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            python_type = cls._json_type_to_python(prop_schema)
            description = prop_schema.get("description", "")

            if prop_name in required:
                fields[prop_name] = (python_type, Field(description=description))
            else:
                fields[prop_name] = (
                    python_type | None,
                    Field(default=None, description=description),
                )

        model_name = f"{tool_name.title().replace('_', '')}Args"
        return create_model(model_name, **fields)

    @classmethod
    def _json_type_to_python(cls, prop_schema: dict[str, Any]) -> type:
        """Convert JSON schema type to Python type.

        Args:
            prop_schema: Property schema.

        Returns:
            Python type.
        """
        json_type = prop_schema.get("type", "string")

        type_mapping: dict[str, type] = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        return type_mapping.get(json_type, str)


class _MCPToolInstance(BaseTool):
    """Internal tool instance for MCP tools.

    This class wraps MCP tool execution with proper connection management.
    """

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        config: MCPServerConfig,
        original_tool_name: str,
        server_name: str,
    ) -> None:
        """Initialize MCP tool instance.

        Args:
            name: Prefixed tool name.
            description: Tool description.
            args_schema: Pydantic model for arguments.
            config: MCP server configuration.
            original_tool_name: Original tool name on MCP server.
            server_name: Server name.
        """
        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema,
        )
        self._config = config
        self._original_tool_name = original_tool_name
        self._server_name = server_name

    @property
    def config(self) -> MCPServerConfig:
        """Get the MCP server configuration."""
        return self._config

    @property
    def original_tool_name(self) -> str:
        """Get the original tool name."""
        return self._original_tool_name

    @property
    def server_name(self) -> str:
        """Get the server name."""
        return self._server_name

    def _run(self, **kwargs: Any) -> str:
        """Execute the MCP tool.

        Args:
            **kwargs: Tool arguments.

        Returns:
            Tool execution result.
        """
        try:
            try:
                asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._run_async(**kwargs)
                    )
                    return future.result()
            except RuntimeError:
                return asyncio.run(self._run_async(**kwargs))
        except Exception as e:
            raise RuntimeError(
                f"Error executing MCP tool {self.original_tool_name}: {e!s}"
            ) from e

    async def _run_async(self, **kwargs: Any) -> str:
        """Async implementation of tool execution.

        Args:
            **kwargs: Tool arguments.

        Returns:
            Tool execution result.
        """
        from crewai.mcp import MCPClient

        transport = MCPTool._create_transport(self._config)
        client = MCPClient(
            transport,
            connect_timeout=MCP_CONNECTION_TIMEOUT,
            execution_timeout=MCP_TOOL_EXECUTION_TIMEOUT,
            discovery_timeout=MCP_DISCOVERY_TIMEOUT,
        )

        try:
            await client.connect()
            result = await client.call_tool(self._original_tool_name, kwargs)

            if isinstance(result, str):
                return result

            if hasattr(result, "content") and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    content_item = result.content[0]
                    if hasattr(content_item, "text"):
                        return str(content_item.text)
                    return str(content_item)
                return str(result.content)

            return str(result)
        finally:
            await client.disconnect()
