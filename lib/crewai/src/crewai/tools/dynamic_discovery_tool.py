"""Dynamic discovery tool that uses a BaseDiscoveryProvider to find and optionally register tools."""

from __future__ import annotations

import asyncio
from typing import Any

from crewai.tools.base_discovery_provider import BaseDiscoveryProvider
from crewai.tools.base_tool import BaseTool


class DynamicDiscoveryTool(BaseTool):
    """A CrewAI tool that runs semantic search via a configured discovery provider.

    When run, it calls the provider's semantic search with the given query,
    resolves results to BaseTool instances, and either returns that list or
    automatically adds them to a configured agent's tools (and returns a summary).
    """

    def __init__(
        self,
        provider: BaseDiscoveryProvider,
        *,
        limit: int = 10,
        agent: Any = None,
        auto_register: bool = True,
        name: str = "dynamic_tool_discovery",
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the dynamic discovery tool.

        Args:
            provider: Discovery provider used for semantic search and resolve.
            limit: Default maximum number of tools to discover per run.
            agent: Optional agent (or any object with a mutable ``tools`` list).
                When set and ``auto_register`` is True, discovered tools are
                appended to ``agent.tools``.
            auto_register: If True and ``agent`` is set, discovered tools are
                registered to ``agent.tools``. Ignored when ``agent`` is None.
            name: Tool name shown to the model.
            description: Tool description. Defaults to a standard discovery description.
            **kwargs: Passed through to BaseTool (e.g. result_as_answer, max_usage_count).
        """
        desc = description or (
            "Use this tool to discover other tools by natural language. "
            "Provide a search_query describing what you need (e.g. 'send Slack messages', "
            "'query a database'). Returns a list of available tools or registers them to the agent."
        )
        super().__init__(
            name=name,
            description=desc,
            **kwargs,
        )
        self._provider = provider
        self._limit = limit
        self._agent = agent
        self._auto_register = auto_register

    def _run(
        self,
        search_query: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Run discovery: search via the provider and optionally register tools to the agent.

        Args:
            search_query: Natural language or keyword query for tool discovery.
            limit: Max number of tools to return. If None, uses the tool's default limit.

        Returns:
            Dict with keys: tools (list of BaseTool), count (int), registered (bool),
            summary (str). The summary is suitable for the model; tools can be used
            programmatically by code that holds a reference to this tool or the agent.
        """
        effective_limit = limit if limit is not None else self._limit
        tools = asyncio.run(
            self._provider.search_semantic_and_resolve(
                search_query,
                limit=effective_limit,
            )
        )
        registered = False
        if (
            self._auto_register
            and self._agent is not None
            and hasattr(self._agent, "tools")
        ):
            agent_tools = self._agent.tools
            if agent_tools is not None:
                agent_tools.extend(tools)
                registered = True

        summary = (
            f"Discovered {len(tools)} tool(s): "
            + ", ".join(t.name for t in tools)
            + (" and registered them to the agent." if registered else ".")
        )
        return {
            "tools": tools,
            "count": len(tools),
            "registered": registered,
            "summary": summary,
        }

    async def _arun(
        self,
        search_query: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Async implementation of discovery and optional registration."""
        effective_limit = limit if limit is not None else self._limit
        tools = await self._provider.search_semantic_and_resolve(
            search_query,
            limit=effective_limit,
        )
        registered = False
        if (
            self._auto_register
            and self._agent is not None
            and hasattr(self._agent, "tools")
        ):
            agent_tools = self._agent.tools
            if agent_tools is not None:
                agent_tools.extend(tools)
                registered = True

        summary = (
            f"Discovered {len(tools)} tool(s): "
            + ", ".join(t.name for t in tools)
            + (" and registered them to the agent." if registered else ".")
        )
        return {
            "tools": tools,
            "count": len(tools),
            "registered": registered,
            "summary": summary,
        }
