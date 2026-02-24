"""MCP tool resolution for CrewAI agents.

This module extracts all MCP-related tool resolution logic from the Agent class
into a standalone MCPToolResolver. It handles three flavours of MCP reference:

  1. Native configs:   MCPServerStdio / MCPServerHTTP / MCPServerSSE objects.
  2. HTTPS URLs:       e.g. "https://mcp.example.com/api"
  3. AMP references:   e.g. "notion" or "notion#search" (legacy "crewai-amp:" prefix also works)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import urlparse

from crewai.mcp.client import MCPClient
from crewai.mcp.config import (
    MCPServerConfig,
    MCPServerHTTP,
    MCPServerSSE,
    MCPServerStdio,
)
from crewai.mcp.transports.http import HTTPTransport
from crewai.mcp.transports.sse import SSETransport
from crewai.mcp.transports.stdio import StdioTransport


if TYPE_CHECKING:
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.logger import Logger

MCP_CONNECTION_TIMEOUT: Final[int] = 10
MCP_TOOL_EXECUTION_TIMEOUT: Final[int] = 30
MCP_DISCOVERY_TIMEOUT: Final[int] = 15
MCP_MAX_RETRIES: Final[int] = 3

_mcp_schema_cache: dict[str, Any] = {}
_cache_ttl: Final[int] = 300  # 5 minutes


class MCPToolResolver:
    """Resolves MCP server references / configs into CrewAI ``BaseTool`` instances.

    Typical lifecycle::

        resolver = MCPToolResolver(agent=my_agent, logger=my_agent._logger)
        tools = resolver.resolve(my_agent.mcps)
        # … agent executes tasks using *tools* …
        resolver.cleanup()

    The resolver owns the MCP client connections it creates and is responsible
    for tearing them down via :meth:`cleanup`.
    """

    def __init__(self, agent: Any, logger: Logger) -> None:
        self._agent = agent
        self._logger = logger
        self._clients: list[Any] = []

    @property
    def clients(self) -> list[Any]:
        return list(self._clients)

    def resolve(self, mcps: list[str | MCPServerConfig]) -> list[BaseTool]:
        """Convert MCP server references/configs to CrewAI tools."""
        all_tools: list[BaseTool] = []
        amp_refs: list[tuple[str, str | None]] = []

        for mcp_config in mcps:
            if isinstance(mcp_config, str) and mcp_config.startswith("https://"):
                all_tools.extend(self._resolve_string(mcp_config))
            elif isinstance(mcp_config, str):
                amp_refs.append(self._parse_amp_ref(mcp_config))
            else:
                tools, client = self._resolve_native(mcp_config)
                all_tools.extend(tools)
                if client:
                    self._clients.append(client)

        if amp_refs:
            tools, clients = self._resolve_amp(amp_refs)
            all_tools.extend(tools)
            self._clients.extend(clients)

        return all_tools

    def cleanup(self) -> None:
        """Disconnect all MCP client connections."""
        if not self._clients:
            return

        async def _disconnect_all() -> None:
            for client in self._clients:
                if client and hasattr(client, "connected") and client.connected:
                    await client.disconnect()

        try:
            asyncio.run(_disconnect_all())
        except Exception as e:
            self._logger.log("error", f"Error during MCP client cleanup: {e}")
        finally:
            self._clients.clear()


    @staticmethod
    def _parse_amp_ref(mcp_config: str) -> tuple[str, str | None]:
        """Parse an AMP reference into *(slug, optional tool name)*.

        Accepts both bare slugs (``"notion"``, ``"notion#search"``) and the
        legacy ``"crewai-amp:notion"`` form.
        """
        bare = mcp_config.removeprefix("crewai-amp:")
        slug, _, specific_tool = bare.partition("#")
        return slug, specific_tool or None

    def _resolve_amp(
        self, amp_refs: list[tuple[str, str | None]]
    ) -> tuple[list[BaseTool], list[Any]]:
        """Fetch AMP configs in bulk and return their tools and clients."""
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.mcp_events import MCPConfigFetchFailedEvent

        unique_slugs = list(dict.fromkeys(slug for slug, _ in amp_refs))
        amp_configs_map = self._fetch_amp_mcp_configs(unique_slugs)

        all_tools: list[BaseTool] = []
        all_clients: list[Any] = []

        for slug, specific_tool in amp_refs:
            config_dict = amp_configs_map.get(slug)
            if not config_dict:
                crewai_event_bus.emit(
                    self,
                    MCPConfigFetchFailedEvent(
                        slug=slug,
                        error=f"Config for '{slug}' not found. Make sure it is connected in your account.",
                        error_type="not_connected",
                    ),
                )
                continue

            mcp_server_config = self._build_mcp_config_from_dict(config_dict)

            if specific_tool:
                from crewai.mcp.filters import create_static_tool_filter

                mcp_server_config.tool_filter = create_static_tool_filter(
                    allowed_tool_names=[specific_tool]
                )

            try:
                tools, client = self._resolve_native(mcp_server_config)
                all_tools.extend(tools)
                if client:
                    all_clients.append(client)
            except Exception as e:
                crewai_event_bus.emit(
                    self,
                    MCPConfigFetchFailedEvent(
                        slug=slug,
                        error=str(e),
                        error_type="connection_failed",
                    ),
                )

        return all_tools, all_clients

    def _fetch_amp_mcp_configs(
        self, slugs: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch MCP server configurations via CrewAI+ API.

        Sends a GET request to the CrewAI+ mcps/configs endpoint with
        comma-separated slugs. CrewAI+ proxies the request to crewai-oauth.

        API-level failures return ``{}``; individual slugs will then
        surface as ``MCPConfigFetchFailedEvent`` in :meth:`_resolve_amp`.
        """
        import requests

        try:
            from crewai_tools.tools.crewai_platform_tools.misc import (
                get_platform_integration_token,
            )

            from crewai.cli.plus_api import PlusAPI

            plus_api = PlusAPI(api_key=get_platform_integration_token())
            response = plus_api.get_mcp_configs(slugs)

            if response.status_code == 200:
                return response.json().get("configs", {})

            self._logger.log(
                "debug",
                f"Failed to fetch MCP configs: HTTP {response.status_code}",
            )
            return {}

        except requests.exceptions.RequestException as e:
            self._logger.log("debug", f"Failed to fetch MCP configs: {e}")
            return {}
        except Exception as e:
            self._logger.log(
                "debug", f"Cannot fetch AMP MCP configs: {e}"
            )
            return {}

    def _resolve_string(self, mcp_ref: str) -> list[BaseTool]:
        """Resolve a plain string MCP reference (currently only HTTPS URLs)."""
        if mcp_ref.startswith("https://"):
            return self._resolve_external(mcp_ref)
        return []

    def _resolve_external(self, mcp_ref: str) -> list[BaseTool]:
        """Resolve an HTTPS MCP server URL into tools."""
        from crewai.tools.mcp_tool_wrapper import MCPToolWrapper

        if "#" in mcp_ref:
            server_url, specific_tool = mcp_ref.split("#", 1)
        else:
            server_url, specific_tool = mcp_ref, None

        server_params = {"url": server_url}
        server_name = self._extract_server_name(server_url)

        try:
            tool_schemas = self._get_mcp_tool_schemas(server_params)

            if not tool_schemas:
                self._logger.log(
                    "warning", f"No tools discovered from MCP server: {server_url}"
                )
                return []

            tools = []
            for tool_name, schema in tool_schemas.items():
                if specific_tool and tool_name != specific_tool:
                    continue

                try:
                    wrapper = MCPToolWrapper(
                        mcp_server_params=server_params,
                        tool_name=tool_name,
                        tool_schema=schema,
                        server_name=server_name,
                    )
                    tools.append(wrapper)
                except Exception as e:
                    self._logger.log(
                        "warning",
                        f"Failed to create MCP tool wrapper for {tool_name}: {e}",
                    )
                    continue

            if specific_tool and not tools:
                self._logger.log(
                    "warning",
                    f"Specific tool '{specific_tool}' not found on MCP server: {server_url}",
                )

            return cast(list[BaseTool], tools)

        except Exception as e:
            self._logger.log(
                "warning", f"Failed to connect to MCP server {server_url}: {e}"
            )
            return []

    def _resolve_native(
        self, mcp_config: MCPServerConfig
    ) -> tuple[list[BaseTool], Any | None]:
        """Resolve an ``MCPServerConfig`` into tools, returning the client for cleanup."""
        from crewai.tools.base_tool import BaseTool
        from crewai.tools.mcp_native_tool import MCPNativeTool

        transport: StdioTransport | HTTPTransport | SSETransport
        if isinstance(mcp_config, MCPServerStdio):
            transport = StdioTransport(
                command=mcp_config.command,
                args=mcp_config.args,
                env=mcp_config.env,
            )
            server_name = f"{mcp_config.command}_{'_'.join(mcp_config.args)}"
        elif isinstance(mcp_config, MCPServerHTTP):
            transport = HTTPTransport(
                url=mcp_config.url,
                headers=mcp_config.headers,
                streamable=mcp_config.streamable,
            )
            server_name = self._extract_server_name(mcp_config.url)
        elif isinstance(mcp_config, MCPServerSSE):
            transport = SSETransport(
                url=mcp_config.url,
                headers=mcp_config.headers,
            )
            server_name = self._extract_server_name(mcp_config.url)
        else:
            raise ValueError(f"Unsupported MCP server config type: {type(mcp_config)}")

        client = MCPClient(
            transport=transport,
            cache_tools_list=mcp_config.cache_tools_list,
        )

        async def _setup_client_and_list_tools() -> list[dict[str, Any]]:
            try:
                if not client.connected:
                    await client.connect()

                tools_list = await client.list_tools()

                try:
                    await client.disconnect()
                    await asyncio.sleep(0.1)
                except Exception as e:
                    self._logger.log("error", f"Error during disconnect: {e}")

                return tools_list
            except Exception as e:
                if client.connected:
                    await client.disconnect()
                    await asyncio.sleep(0.1)
                raise RuntimeError(
                    f"Error during setup client and list tools: {e}"
                ) from e

        try:
            try:
                asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, _setup_client_and_list_tools()
                    )
                    tools_list = future.result()
            except RuntimeError:
                try:
                    tools_list = asyncio.run(_setup_client_and_list_tools())
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "cancel scope" in error_msg or "task" in error_msg:
                        raise ConnectionError(
                            "MCP connection failed due to event loop cleanup issues. "
                            "This may be due to authentication errors or server unavailability."
                        ) from e
                except asyncio.CancelledError as e:
                    raise ConnectionError(
                        "MCP connection was cancelled. This may indicate an authentication "
                        "error or server unavailability."
                    ) from e

            if mcp_config.tool_filter:
                filtered_tools = []
                for tool in tools_list:
                    if callable(mcp_config.tool_filter):
                        try:
                            from crewai.mcp.filters import ToolFilterContext

                            context = ToolFilterContext(
                                agent=self._agent,
                                server_name=server_name,
                                run_context=None,
                            )
                            if mcp_config.tool_filter(context, tool):  # type: ignore[call-arg, arg-type]
                                filtered_tools.append(tool)
                        except (TypeError, AttributeError):
                            if mcp_config.tool_filter(tool):  # type: ignore[call-arg, arg-type]
                                filtered_tools.append(tool)
                    else:
                        filtered_tools.append(tool)
                tools_list = filtered_tools

            tools = []
            for tool_def in tools_list:
                tool_name = tool_def.get("name", "")
                original_tool_name = tool_def.get("original_name", tool_name)
                if not tool_name:
                    continue

                args_schema = None
                if tool_def.get("inputSchema"):
                    args_schema = self._json_schema_to_pydantic(
                        tool_name, tool_def["inputSchema"]
                    )

                tool_schema = {
                    "description": tool_def.get("description", ""),
                    "args_schema": args_schema,
                }

                try:
                    native_tool = MCPNativeTool(
                        mcp_client=client,
                        tool_name=tool_name,
                        tool_schema=tool_schema,
                        server_name=server_name,
                        original_tool_name=original_tool_name,
                    )
                    tools.append(native_tool)
                except Exception as e:
                    self._logger.log("error", f"Failed to create native MCP tool: {e}")
                    continue

            return cast(list[BaseTool], tools), client
        except Exception as e:
            if client.connected:
                asyncio.run(client.disconnect())

            raise RuntimeError(f"Failed to get native MCP tools: {e}") from e

    @staticmethod
    def _build_mcp_config_from_dict(
        config_dict: dict[str, Any],
    ) -> MCPServerConfig:
        """Convert a config dict from crewai-oauth into an MCPServerConfig."""
        config_type = config_dict.get("type", "http")

        if config_type == "sse":
            return MCPServerSSE(
                url=config_dict["url"],
                headers=config_dict.get("headers"),
                cache_tools_list=config_dict.get("cache_tools_list", False),
            )

        return MCPServerHTTP(
            url=config_dict["url"],
            headers=config_dict.get("headers"),
            streamable=config_dict.get("streamable", True),
            cache_tools_list=config_dict.get("cache_tools_list", False),
        )

    @staticmethod
    def _extract_server_name(server_url: str) -> str:
        """Extract clean server name from URL for tool prefixing."""
        parsed = urlparse(server_url)
        domain = parsed.netloc.replace(".", "_")
        path = parsed.path.replace("/", "_").strip("_")
        return f"{domain}_{path}" if path else domain

    def _get_mcp_tool_schemas(
        self, server_params: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Get tool schemas from MCP server with caching."""
        server_url = server_params["url"]

        cache_key = server_url
        current_time = time.time()

        if cache_key in _mcp_schema_cache:
            cached_data, cache_time = _mcp_schema_cache[cache_key]
            if current_time - cache_time < _cache_ttl:
                self._logger.log(
                    "debug", f"Using cached MCP tool schemas for {server_url}"
                )
                return cached_data  # type: ignore[no-any-return]

        try:
            schemas = asyncio.run(self._get_mcp_tool_schemas_async(server_params))
            _mcp_schema_cache[cache_key] = (schemas, current_time)
            return schemas
        except Exception as e:
            self._logger.log(
                "warning", f"Failed to get MCP tool schemas from {server_url}: {e}"
            )
            return {}

    async def _get_mcp_tool_schemas_async(
        self, server_params: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Async implementation of MCP tool schema retrieval."""
        server_url = server_params["url"]
        return await self._retry_mcp_discovery(
            self._discover_mcp_tools_with_timeout, server_url
        )

    async def _retry_mcp_discovery(
        self, operation_func: Any, server_url: str
    ) -> dict[str, dict[str, Any]]:
        """Retry MCP discovery with exponential backoff."""
        last_error = None

        for attempt in range(MCP_MAX_RETRIES):
            result, error, should_retry = await self._attempt_mcp_discovery(
                operation_func, server_url
            )

            if result is not None:
                return result

            if not should_retry:
                raise RuntimeError(error)

            last_error = error
            if attempt < MCP_MAX_RETRIES - 1:
                wait_time = 2**attempt
                await asyncio.sleep(wait_time)

        raise RuntimeError(
            f"Failed to discover MCP tools after {MCP_MAX_RETRIES} attempts: {last_error}"
        )

    @staticmethod
    async def _attempt_mcp_discovery(
        operation_func: Any, server_url: str
    ) -> tuple[dict[str, dict[str, Any]] | None, str, bool]:
        """Attempt single MCP discovery; returns *(result, error_message, should_retry)*."""
        try:
            result = await operation_func(server_url)
            return result, "", False

        except ImportError:
            return (
                None,
                "MCP library not available. Please install with: pip install mcp",
                False,
            )

        except asyncio.TimeoutError:
            return (
                None,
                f"MCP discovery timed out after {MCP_DISCOVERY_TIMEOUT} seconds",
                True,
            )

        except Exception as e:
            error_str = str(e).lower()

            if "authentication" in error_str or "unauthorized" in error_str:
                return None, f"Authentication failed for MCP server: {e!s}", False
            if "connection" in error_str or "network" in error_str:
                return None, f"Network connection failed: {e!s}", True
            if "json" in error_str or "parsing" in error_str:
                return None, f"Server response parsing error: {e!s}", True
            return None, f"MCP discovery error: {e!s}", False

    async def _discover_mcp_tools_with_timeout(
        self, server_url: str
    ) -> dict[str, dict[str, Any]]:
        """Discover MCP tools with timeout wrapper."""
        return await asyncio.wait_for(
            self._discover_mcp_tools(server_url), timeout=MCP_DISCOVERY_TIMEOUT
        )

    async def _discover_mcp_tools(self, server_url: str) -> dict[str, dict[str, Any]]:
        """Discover tools from an MCP server (HTTPS / streamable-HTTP path)."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        from crewai.utilities.string_utils import sanitize_tool_name

        async with streamablehttp_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(
                    session.initialize(), timeout=MCP_CONNECTION_TIMEOUT
                )

                tools_result = await asyncio.wait_for(
                    session.list_tools(),
                    timeout=MCP_DISCOVERY_TIMEOUT - MCP_CONNECTION_TIMEOUT,
                )

                schemas = {}
                for tool in tools_result.tools:
                    args_schema = None
                    if hasattr(tool, "inputSchema") and tool.inputSchema:
                        args_schema = self._json_schema_to_pydantic(
                            sanitize_tool_name(tool.name), tool.inputSchema
                        )

                    schemas[sanitize_tool_name(tool.name)] = {
                        "description": getattr(tool, "description", ""),
                        "args_schema": args_schema,
                    }
                return schemas

    @staticmethod
    def _json_schema_to_pydantic(
        tool_name: str, json_schema: dict[str, Any]
    ) -> type:
        """Convert JSON Schema to a Pydantic model for tool arguments."""
        from crewai.utilities.pydantic_schema_utils import create_model_from_schema

        model_name = f"{tool_name.replace('-', '_').replace(' ', '_')}Schema"
        return create_model_from_schema(
            json_schema,
            model_name=model_name,
            enrich_descriptions=True,
        )
