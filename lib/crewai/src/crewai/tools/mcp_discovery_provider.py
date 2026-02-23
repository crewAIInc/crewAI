"""Discovery provider backed by yksanjo/mcp-discovery.

This provider uses the public HTTP API exposed by the mcp-discovery
project to perform **semantic search over MCP servers**, then exposes
the results through the common ``BaseDiscoveryProvider`` interface.

It also knows how to turn a ``DiscoveryEntry`` into a concrete
``MCPToolWrapper`` instance, so the discovered MCP servers (and their
tools) can be used as CrewAI tools.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import httpx

from crewai.tools.base_discovery_provider import BaseDiscoveryProvider, DiscoveryEntry
from crewai.tools.base_tool import BaseTool
from crewai.tools.mcp_tool_wrapper import MCPToolWrapper


DEFAULT_MCP_DISCOVERY_BASE_URL = "https://mcp-discovery-two.vercel.app"


class MCPDiscoveryProvider(BaseDiscoveryProvider):
    """Discovery provider that talks to the mcp-discovery HTTP API.

    The core semantic search is delegated to the mcp-discovery service,
    which indexes MCP servers and returns ranked recommendations.

    Parameters
    ----------
    base_url:
        Base URL of the mcp-discovery API. If not provided, it will use
        the ``MCP_DISCOVERY_API_BASE_URL`` environment variable, falling
        back to the public deployment at
        ``https://mcp-discovery-two.vercel.app``.
    timeout:
        HTTP timeout in seconds for discovery requests.
    server_url_resolver:
        Optional callable to turn a ``DiscoveryEntry`` into an MCP HTTP
        server URL. If not provided, ``resolve_to_tool`` will fall back
        to ``entry.source_uri`` and raise a ``ValueError`` if it is not
        set. This hook allows callers to customize how a discovered
        server maps to a concrete MCP HTTP endpoint.
    default_tool_name:
        Default MCP tool name to wrap on the discovered server if no
        specific tool is provided in ``DiscoveryEntry.raw_metadata``.
        This is a pragmatic default; real integrations will typically
        populate a concrete tool name via an additional lookup step.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 10.0,
        server_url_resolver: Callable[[DiscoveryEntry], str] | None = None,
        default_tool_name: str = "default",
    ) -> None:
        self._base_url = (
            base_url
            or os.getenv("MCP_DISCOVERY_API_BASE_URL")
            or DEFAULT_MCP_DISCOVERY_BASE_URL
        ).rstrip("/")
        self._timeout = timeout
        self._server_url_resolver = server_url_resolver
        self._default_tool_name = default_tool_name

    # ------------------------------------------------------------------
    # BaseDiscoveryProvider implementation
    # ------------------------------------------------------------------

    @property
    def provider_id(self) -> str:
        return "mcp-discovery"

    async def search_semantic(
        self,
        query: str,
        *,
        limit: int = 10,
        **_: Any,
    ) -> list[DiscoveryEntry]:
        """Use the mcp-discovery API to find relevant MCP servers.

        This method calls the public HTTP endpoint exposed by
        mcp-discovery. The current API (see yksanjo/mcp-discovery and
        mcp-discovery-api docs) expects a JSON body compatible with the
        ``DiscoverInput`` type:

        .. code-block:: json

            {
              "need": "<natural language requirement>",
              "limit": <max_results>
            }

        and responds with a ``DiscoverOutput`` object containing a
        ``recommendations`` array of ``ServerRecommendation`` items.
        Each recommendation is normalized into a ``DiscoveryEntry``.
        """
        payload = {
            "need": query,
            "limit": limit,
        }

        url = f"{self._base_url}/api/v1/discover"

        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        recommendations = data.get("recommendations") or []

        entries: list[DiscoveryEntry] = []
        for rec in recommendations:
            # ServerRecommendation shape from mcp-discovery:
            # {
            #   server: string;
            #   npm_package: string | null;
            #   install_command: string;
            #   confidence: number;
            #   description: string | null;
            #   capabilities: string[];
            #   metrics: { ... };
            #   docs_url: string | null;
            #   github_url: string | null;
            # }
            server_name = rec.get("server") or ""
            if not server_name:
                # Skip malformed recommendation entries.
                continue

            entry_id = server_name

            description = rec.get("description") or ""
            source_uri = rec.get("docs_url") or rec.get("github_url")

            raw_metadata: dict[str, Any] = {
                "npm_package": rec.get("npm_package"),
                "install_command": rec.get("install_command"),
                "confidence": rec.get("confidence"),
                "capabilities": rec.get("capabilities") or [],
                "metrics": rec.get("metrics") or {},
                "docs_url": rec.get("docs_url"),
                "github_url": rec.get("github_url"),
            }

            entries.append(
                DiscoveryEntry(
                    id=entry_id,
                    name=server_name,
                    description=description,
                    source_uri=source_uri,
                    input_schema=None,  # server-level; tool schemas require a separate MCP list_tools step
                    provider_id=self.provider_id,
                    raw_metadata=raw_metadata,
                )
            )

        return entries

    async def resolve_to_tool(self, entry: DiscoveryEntry) -> BaseTool:
        """Create an ``MCPToolWrapper`` for a discovered MCP server.

        Note
        ----
        mcp-discovery itself returns **server-level** metadata
        (documentation URLs, install commands, capabilities, metrics),
        not individual MCP tool schemas.

        This implementation therefore makes two pragmatic assumptions:

        1. There exists an HTTP-accessible MCP endpoint for the server,
           whose URL can either be derived from the ``DiscoveryEntry``
           (via ``source_uri``) or supplied by a custom
           ``server_url_resolver`` callable.
        2. A reasonable default tool name can be used if the caller has
           not pre-populated a more specific MCP tool name in
           ``entry.raw_metadata['tool_name']``.

        Real-world integrations will typically add an extra discovery
        step that, given a server URL, lists its MCP tools and enriches
        ``DiscoveryEntry.raw_metadata`` with concrete tool-level
        schemas.
        """
        # Determine MCP server URL
        server_url: str | None
        if self._server_url_resolver is not None:
            server_url = self._server_url_resolver(entry)
        else:
            server_url = entry.source_uri

        if not server_url:
            raise ValueError(
                "Cannot resolve MCP server URL for discovery entry "
                f"{entry.id!r}; provide source_uri or a server_url_resolver."
            )

        mcp_server_params = {"url": server_url}

        # Tool name preference: explicit in metadata -> default from provider
        tool_name = entry.raw_metadata.get("tool_name") or self._default_tool_name

        # Basic tool schema: description + optional args_schema from metadata
        tool_schema: dict[str, Any] = {
            "description": entry.description
            or f"Tool {tool_name} discovered via mcp-discovery for server {entry.name}",
        }

        if "args_schema" in entry.raw_metadata:
            tool_schema["args_schema"] = entry.raw_metadata["args_schema"]

        # Use the server's display name as the MCPToolWrapper server_name prefix
        server_name = entry.name

        return MCPToolWrapper(
            mcp_server_params=mcp_server_params,
            tool_name=tool_name,
            tool_schema=tool_schema,
            server_name=server_name,
        )
