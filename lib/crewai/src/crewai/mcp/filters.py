"""Tool filtering support for MCP servers.

This module provides utilities for filtering tools from MCP servers,
including static allow/block lists, dynamic context-aware filtering, and
semantic (embedding-based) filtering.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from crewai.rag.core.base_embeddings_callable import EmbeddingFunction


class ToolFilterContext(BaseModel):
    """Context for dynamic tool filtering.

    This context is passed to dynamic tool filters to provide
    information about the agent, run context, and server.
    """

    agent: Any = Field(..., description="The agent requesting tools.")
    server_name: str = Field(..., description="Name of the MCP server.")
    run_context: dict[str, Any] | None = Field(
        default=None,
        description="Optional run context for additional filtering logic.",
    )


ToolFilter = (
    Callable[[ToolFilterContext, dict[str, Any]], bool]
    | Callable[[dict[str, Any]], bool]
)


class StaticToolFilter:
    """Static tool filter with allow/block lists.

    This filter provides simple allow/block list filtering based on
    tool names. Useful for restricting which tools are available
    from an MCP server.

    Example:
        ```python
        filter = StaticToolFilter(
            allowed_tool_names=["read_file", "write_file"],
            blocked_tool_names=["delete_file"],
        )
        ```
    """

    def __init__(
        self,
        allowed_tool_names: list[str] | None = None,
        blocked_tool_names: list[str] | None = None,
    ) -> None:
        """Initialize static tool filter.

        Args:
            allowed_tool_names: List of tool names to allow. If None,
                all tools are allowed (unless blocked).
            blocked_tool_names: List of tool names to block. Blocked tools
                take precedence over allowed tools.
        """
        self.allowed_tool_names = set(allowed_tool_names or [])
        self.blocked_tool_names = set(blocked_tool_names or [])

    def __call__(self, tool: dict[str, Any]) -> bool:
        """Filter tool based on allow/block lists.

        Args:
            tool: Tool definition dictionary with at least 'name' key.

        Returns:
            True if tool should be included, False otherwise.
        """
        tool_name = tool.get("name", "")

        # Blocked tools take precedence over allowed tools
        if self.blocked_tool_names and tool_name in self.blocked_tool_names:
            return False

        if self.allowed_tool_names:
            return tool_name in self.allowed_tool_names

        return True


def create_static_tool_filter(
    allowed_tool_names: list[str] | None = None,
    blocked_tool_names: list[str] | None = None,
) -> Callable[[dict[str, Any]], bool]:
    """Create a static tool filter function.

    This is a convenience function for creating static tool filters
    with allow/block lists.

    Args:
        allowed_tool_names: List of tool names to allow. If None,
            all tools are allowed (unless blocked).
        blocked_tool_names: List of tool names to block. Blocked tools
            take precedence over allowed tools.

    Returns:
        Tool filter function that returns True for allowed tools.

    Example:
        ```python
        filter_fn = create_static_tool_filter(
            allowed_tool_names=["read_file", "write_file"],
            blocked_tool_names=["delete_file"],
        )

        # Use in MCPServerStdio
        mcp_server = MCPServerStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            tool_filter=filter_fn,
        )
        ```
    """
    return StaticToolFilter(
        allowed_tool_names=allowed_tool_names,
        blocked_tool_names=blocked_tool_names,
    )


def create_dynamic_tool_filter(
    filter_func: Callable[[ToolFilterContext, dict[str, Any]], bool],
) -> Callable[[ToolFilterContext, dict[str, Any]], bool]:
    """Create a dynamic tool filter function.

    This function wraps a dynamic filter function that has access
    to the tool filter context (agent, server, run context).

    Args:
        filter_func: Function that takes (context, tool) and returns bool.

    Returns:
        Tool filter function that can be used with MCP server configs.

    Example:
        ```python
        async def context_aware_filter(
            context: ToolFilterContext, tool: dict[str, Any]
        ) -> bool:
            # Block dangerous tools for code reviewers
            if context.agent.role == "Code Reviewer":
                if tool["name"].startswith("danger_"):
                    return False
            return True


        filter_fn = create_dynamic_tool_filter(context_aware_filter)

        mcp_server = MCPServerStdio(
            command="python", args=["server.py"], tool_filter=filter_fn
        )
        ```
    """
    return filter_func


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (pure Python, no numpy).

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1]; 0.0 if either vector is empty or zero.
    """
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticToolFilter:
    """Dynamic MCP tool filter that includes tools semantically relevant to the agent.

    Embeds a query derived from the requesting agent's role/goal/backstory
    (or from ``run_context[run_context_key]`` when populated) and each tool's
    ``name`` + ``description``, including the tool only when cosine similarity
    between the two embeddings is at least ``threshold``.

    Fits the existing per-tool ``ToolFilter`` protocol and is applied to MCP
    server tools at resolution time (see ``tool_resolver``). It is
    threshold-based (not top-K) because the protocol decides each tool
    independently.

    Fails open: if no query can be derived, or the embedder raises, the tool is
    included — a misconfigured embedder never silently strips all tools.

    Example:
        ```python
        from crewai.rag.embeddings.factory import build_embedder
        from crewai.mcp.filters import SemanticToolFilter

        embedder = build_embedder({"provider": "openai", "config": {...}})
        mcp_server = MCPServerStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            tool_filter=SemanticToolFilter(embedder=embedder, threshold=0.35),
        )
        ```
    """

    def __init__(
        self,
        embedder: EmbeddingFunction,
        threshold: float = 0.3,
        run_context_key: str = "query",
    ) -> None:
        """Initialize semantic tool filter.

        Args:
            embedder: Embedding function used to embed the query and tool texts.
                Build one with ``crewai.rag.embeddings.factory.build_embedder``.
            threshold: Minimum cosine similarity (0.0-1.0) for a tool to be
                included.
            run_context_key: Key in ``ToolFilterContext.run_context`` holding an
                explicit query string. When absent, the agent's
                role/goal/backstory is used as the query.
        """
        self._embedder = embedder
        self.threshold = threshold
        self._run_context_key = run_context_key
        self._cache: dict[str, list[float]] = {}

    def __call__(self, context: ToolFilterContext, tool: dict[str, Any]) -> bool:
        """Include ``tool`` if it is semantically relevant to the query.

        Args:
            context: Tool filter context (agent, server, run context).
            tool: Tool definition dict with at least ``name``/``description``.

        Returns:
            True if the tool should be included, False otherwise. Fails open
            (returns True) when no query is available or embedding fails.
        """
        query = self._query_text(context)
        if not query:
            return True

        tool_text = self._tool_text(tool)
        if not tool_text:
            return True

        try:
            query_vec = self._embed(query)
            tool_vec = self._embed(tool_text)
        except Exception:
            return True

        return _cosine_similarity(query_vec, tool_vec) >= self.threshold

    def _query_text(self, context: ToolFilterContext) -> str | None:
        """Derive the query string from run_context or the agent profile."""
        run_context = context.run_context or {}
        query = run_context.get(self._run_context_key)
        if isinstance(query, str) and query.strip():
            return query

        agent = context.agent
        if agent is None:
            return None

        parts = [
            getattr(agent, "role", ""),
            getattr(agent, "goal", ""),
            getattr(agent, "backstory", ""),
        ]
        parts = [p for p in parts if isinstance(p, str) and p.strip()]
        return " ".join(parts) if parts else None

    @staticmethod
    def _tool_text(tool: dict[str, Any]) -> str:
        """Build the text representation of a tool for embedding."""
        name = tool.get("name", "") or ""
        description = tool.get("description", "") or ""
        if not isinstance(name, str):
            name = str(name)
        if not isinstance(description, str):
            description = str(description)
        name = name.strip()
        description = description.strip()
        if not name and not description:
            return ""
        if name and description:
            return f"{name}: {description}"
        return name or description

    def _embed(self, text: str) -> list[float]:
        """Embed ``text`` using the embedder, with per-text caching."""
        if text in self._cache:
            return self._cache[text]
        result = self._embedder(input=[text])
        vec = list(result[0]) if result else []
        self._cache[text] = vec
        return vec


def create_semantic_tool_filter(
    embedder: EmbeddingFunction,
    threshold: float = 0.3,
    run_context_key: str = "query",
) -> SemanticToolFilter:
    """Create a semantic tool filter.

    Convenience wrapper around :class:`SemanticToolFilter`.

    Args:
        embedder: Embedding function (build one with ``build_embedder``).
        threshold: Minimum cosine similarity for inclusion.
        run_context_key: Optional run_context key holding an explicit query.

    Returns:
        A ``SemanticToolFilter`` usable as ``tool_filter`` on an MCP server.
    """
    return SemanticToolFilter(
        embedder=embedder,
        threshold=threshold,
        run_context_key=run_context_key,
    )
