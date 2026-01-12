"""Tool Search Tool for on-demand tool discovery.

This module implements a Tool Search Tool that allows agents to dynamically
discover and load tools on-demand, reducing token consumption when working
with large tool libraries.

Inspired by Anthropic's Tool Search Tool approach for on-demand tool loading.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.pydantic_schema_utils import generate_model_description


class SearchStrategy(str, Enum):
    """Search strategy for tool discovery."""

    KEYWORD = "keyword"
    REGEX = "regex"


class ToolSearchResult(BaseModel):
    """Result from a tool search operation."""

    name: str = Field(description="The name of the tool")
    description: str = Field(description="The description of the tool")
    args_schema: dict[str, Any] = Field(
        description="The JSON schema for the tool's arguments"
    )


class ToolSearchToolSchema(BaseModel):
    """Schema for the Tool Search Tool arguments."""

    query: str = Field(
        description="The search query to find relevant tools. Use keywords that describe the capability you need."
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of tools to return. Default is 5.",
        ge=1,
        le=20,
    )


class ToolSearchTool(BaseTool):
    """A tool that searches through a catalog of tools to find relevant ones.

    This tool enables on-demand tool discovery, allowing agents to work with
    large tool libraries without loading all tool definitions upfront. Instead
    of consuming tokens with all tool definitions, the agent can search for
    relevant tools when needed.

    Example:
        ```python
        from crewai.tools import BaseTool, ToolSearchTool

        # Create your tools
        search_tool = MySearchTool()
        scrape_tool = MyScrapeWebsiteTool()
        database_tool = MyDatabaseTool()

        # Create a tool search tool with your tool catalog
        tool_search = ToolSearchTool(
            tool_catalog=[search_tool, scrape_tool, database_tool],
            search_strategy=SearchStrategy.KEYWORD,
        )

        # Use with an agent - only the tool_search is loaded initially
        agent = Agent(
            role="Researcher",
            tools=[tool_search],  # Other tools discovered on-demand
        )
        ```

    Attributes:
        tool_catalog: List of tools available for search.
        search_strategy: Strategy to use for searching (keyword or regex).
        custom_search_fn: Optional custom search function for advanced matching.
    """

    name: str = Field(
        default="Tool Search",
        description="The name of the tool search tool.",
    )
    description: str = Field(
        default="Search for available tools by describing the capability you need. Returns tool definitions that match your query.",
        description="Description of what the tool search tool does.",
    )
    args_schema: type[BaseModel] = Field(
        default=ToolSearchToolSchema,
        description="The schema for the tool search arguments.",
    )
    tool_catalog: list[BaseTool | CrewStructuredTool] = Field(
        default_factory=list,
        description="List of tools available for search.",
    )
    search_strategy: SearchStrategy = Field(
        default=SearchStrategy.KEYWORD,
        description="Strategy to use for searching tools.",
    )
    custom_search_fn: Callable[
        [str, Sequence[BaseTool | CrewStructuredTool]], list[BaseTool | CrewStructuredTool]
    ] | None = Field(
        default=None,
        description="Optional custom search function for advanced matching.",
    )

    def _run(self, query: str, max_results: int = 5) -> str:
        """Search for tools matching the query.

        Args:
            query: The search query to find relevant tools.
            max_results: Maximum number of tools to return.

        Returns:
            JSON string containing the matching tool definitions.
        """
        if not self.tool_catalog:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No tools available in the catalog.",
                    "tools": [],
                }
            )

        if self.custom_search_fn:
            matching_tools = self.custom_search_fn(query, self.tool_catalog)
        elif self.search_strategy == SearchStrategy.REGEX:
            matching_tools = self._regex_search(query)
        else:
            matching_tools = self._keyword_search(query)

        matching_tools = matching_tools[:max_results]

        if not matching_tools:
            return json.dumps(
                {
                    "status": "no_results",
                    "message": f"No tools found matching query: '{query}'. Try different keywords.",
                    "tools": [],
                }
            )

        tool_results = []
        for tool in matching_tools:
            tool_info = self._get_tool_info(tool)
            tool_results.append(tool_info)

        return json.dumps(
            {
                "status": "success",
                "message": f"Found {len(tool_results)} tool(s) matching your query.",
                "tools": tool_results,
            },
            indent=2,
        )

    def _keyword_search(
        self, query: str
    ) -> list[BaseTool | CrewStructuredTool]:
        """Search tools using keyword matching.

        Args:
            query: The search query.

        Returns:
            List of matching tools sorted by relevance.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_tools: list[tuple[float, BaseTool | CrewStructuredTool]] = []

        for tool in self.tool_catalog:
            score = self._calculate_keyword_score(tool, query_lower, query_words)
            if score > 0:
                scored_tools.append((score, tool))

        scored_tools.sort(key=lambda x: x[0], reverse=True)
        return [tool for _, tool in scored_tools]

    def _calculate_keyword_score(
        self,
        tool: BaseTool | CrewStructuredTool,
        query_lower: str,
        query_words: set[str],
    ) -> float:
        """Calculate relevance score for a tool based on keyword matching.

        Args:
            tool: The tool to score.
            query_lower: Lowercase query string.
            query_words: Set of query words.

        Returns:
            Relevance score (higher is better).
        """
        score = 0.0
        tool_name_lower = tool.name.lower()
        tool_desc_lower = tool.description.lower()

        if query_lower in tool_name_lower:
            score += 10.0
        if query_lower in tool_desc_lower:
            score += 5.0

        for word in query_words:
            if len(word) < 2:
                continue
            if word in tool_name_lower:
                score += 3.0
            if word in tool_desc_lower:
                score += 1.0

        return score

    def _regex_search(
        self, query: str
    ) -> list[BaseTool | CrewStructuredTool]:
        """Search tools using regex pattern matching.

        Args:
            query: The regex pattern to search for.

        Returns:
            List of matching tools.
        """
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        return [
            tool
            for tool in self.tool_catalog
            if pattern.search(tool.name) or pattern.search(tool.description)
        ]

    def _get_tool_info(self, tool: BaseTool | CrewStructuredTool) -> dict[str, Any]:
        """Get tool information as a dictionary.

        Args:
            tool: The tool to get information from.

        Returns:
            Dictionary containing tool name, description, and args schema.
        """
        if isinstance(tool, BaseTool):
            schema_dict = generate_model_description(tool.args_schema)
            args_schema = schema_dict.get("json_schema", {}).get("schema", {})
        else:
            args_schema = tool.args_schema.model_json_schema()

        return {
            "name": tool.name,
            "description": self._get_original_description(tool),
            "args_schema": args_schema,
        }

    def _get_original_description(self, tool: BaseTool | CrewStructuredTool) -> str:
        """Get the original description of a tool without the generated schema.

        Args:
            tool: The tool to get the description from.

        Returns:
            The original tool description.
        """
        description = tool.description
        if "Tool Description:" in description:
            parts = description.split("Tool Description:")
            if len(parts) > 1:
                return parts[1].strip()
        return description

    def add_tool(self, tool: BaseTool | CrewStructuredTool) -> None:
        """Add a tool to the catalog.

        Args:
            tool: The tool to add.
        """
        self.tool_catalog.append(tool)

    def add_tools(self, tools: Sequence[BaseTool | CrewStructuredTool]) -> None:
        """Add multiple tools to the catalog.

        Args:
            tools: The tools to add.
        """
        self.tool_catalog.extend(tools)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the catalog by name.

        Args:
            tool_name: The name of the tool to remove.

        Returns:
            True if the tool was removed, False if not found.
        """
        for i, tool in enumerate(self.tool_catalog):
            if tool.name == tool_name:
                self.tool_catalog.pop(i)
                return True
        return False

    def get_catalog_size(self) -> int:
        """Get the number of tools in the catalog.

        Returns:
            The number of tools in the catalog.
        """
        return len(self.tool_catalog)

    def list_tool_names(self) -> list[str]:
        """List all tool names in the catalog.

        Returns:
            List of tool names.
        """
        return [tool.name for tool in self.tool_catalog]
