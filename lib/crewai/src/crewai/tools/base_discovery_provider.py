"""Abstract interface for tool discovery providers.

Supports multiple backends (e.g. mcp-discovery API, list.agenium.net with agent://)
through a unified semantic search API and conversion to CrewAI BaseTool instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool


# ---------------------------------------------------------------------------
# Discovery result model (provider-agnostic)
# ---------------------------------------------------------------------------


class DiscoveryEntry(BaseModel):
    """Provider-agnostic representation of a discovered tool.

    All discovery backends (mcp-discovery, list.agenium.net, etc.) should
    normalize their results to this shape. The provider then uses
    raw_metadata (and optionally source_uri) to turn the entry into a
    concrete BaseTool (e.g. MCPToolWrapper or an agent://-based tool).
    """

    id: str = Field(
        description="Unique identifier for this tool within the discovery source."
    )
    name: str = Field(
        description="Tool name suitable for display and for BaseTool.name."
    )
    description: str = Field(
        default="",
        description="Tool description for model use and BaseTool.description.",
    )
    source_uri: str | None = Field(
        default=None,
        description="Canonical source (e.g. agent://..., https://..., mcp server URL).",
    )
    input_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema for tool arguments (e.g. MCP inputSchema).",
    )
    provider_id: str = Field(
        description="Identifier of the provider that returned this entry (e.g. 'mcp-discovery', 'agenium').",
    )
    raw_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific data needed to build BaseTool (e.g. mcp_server_params, agent config).",
    )


# ---------------------------------------------------------------------------
# Abstract discovery provider
# ---------------------------------------------------------------------------


class BaseDiscoveryProvider(ABC):
    """Abstract interface for tool discovery services.

    Implementations can back semantic search with different APIs (e.g.
    mcp-discovery, list.agenium.net) and different protocols (MCP, agent://).
    Results are normalized to DiscoveryEntry and can be converted to
    CrewAI BaseTool instances for use in agents.
    """

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider (e.g. 'mcp-discovery', 'agenium')."""
        ...

    @abstractmethod
    async def search_semantic(
        self,
        query: str,
        *,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[DiscoveryEntry]:
        """Run semantic search over the tool registry.

        Args:
            query: Natural language or keyword query describing desired tools.
            limit: Maximum number of entries to return.
            **kwargs: Provider-specific options (e.g. filters, scoring).

        Returns:
            List of discovery entries, ordered by relevance (best first).
        """
        ...

    async def resolve_to_tool(self, entry: DiscoveryEntry) -> BaseTool:
        """Convert a discovery entry into a CrewAI BaseTool instance.

        Default implementation raises NotImplementedError; subclasses must
        implement so that the returned tool can be used with CrewAI agents
        (e.g. MCPToolWrapper for MCP servers, or an agent://-aware wrapper
        for Agenium).

        Args:
            entry: A DiscoveryEntry returned by search_semantic (or list_all).

        Returns:
            A BaseTool instance (e.g. MCPToolWrapper) ready for to_structured_tool().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement resolve_to_tool(DiscoveryEntry) -> BaseTool"
        )

    async def search_semantic_and_resolve(
        self,
        query: str,
        *,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[BaseTool]:
        """Convenience: semantic search then resolve each entry to BaseTool.

        Args:
            query: Semantic search query.
            limit: Max number of tools to return.
            **kwargs: Passed through to search_semantic.

        Returns:
            List of BaseTool instances.
        """
        entries = await self.search_semantic(query, limit=limit, **kwargs)
        tools: list[BaseTool] = []
        for entry in entries:
            try:
                tool = await self.resolve_to_tool(entry)
                tools.append(tool)
            except Exception:  # noqa: S112, PERF203
                # Skip entries that fail to resolve (e.g. unreachable server)
                continue
        return tools

    async def list_all(
        self,
        *,
        limit: int = 100,
        **kwargs: Any,
    ) -> list[DiscoveryEntry]:
        """List available tools without semantic ranking (optional).

        Override if the backend supports listing the full catalog; otherwise
        callers can use search_semantic with a broad query.

        Args:
            limit: Maximum entries to return.
            **kwargs: Provider-specific options.

        Returns:
            List of discovery entries (order is provider-defined).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement list_all; use search_semantic instead."
        )
