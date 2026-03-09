from typing import Any, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class Agent101SearchSchema(BaseModel):
    """Input for Agent101SearchTool."""

    query: str = Field(
        ..., description="Search query to find AI agent tools and services"
    )


class Agent101RecommendSchema(BaseModel):
    """Input for Agent101RecommendTool."""

    task: str = Field(
        ...,
        description="Description of the task you need an AI agent tool for",
    )


class Agent101SearchTool(BaseTool):
    """
    Agent101SearchTool - Search the Agent101 directory for AI agent tools and services.

    Agent101 (https://agent101.ventify.ai) is a directory/yellowpages for AI agents.
    It provides a curated catalog of AI agent tools, APIs, and services across
    categories like search, communication, data, AI services, and more.

    This tool lets you search the directory to discover tools that can extend
    your agent's capabilities.

    No API key required.
    """

    name: str = "Agent101 Search"
    description: str = (
        "Search the Agent101 directory (agent101.ventify.ai) to discover AI agent "
        "tools, APIs, and services. Returns matching entries from the directory."
    )
    args_schema: Type[BaseModel] = Agent101SearchSchema
    base_url: str = "https://agent101.ventify.ai/api"

    def _run(self, **kwargs: Any) -> Any:
        query = kwargs.get("query")
        if not query:
            return "Error: search query is required."

        try:
            response = requests.get(
                f"{self.base_url}/search",
                params={"q": query},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return f"No results found for '{query}'."

            results = []
            for item in data:
                entry = []
                if "name" in item:
                    entry.append(f"Name: {item['name']}")
                if "description" in item:
                    entry.append(f"Description: {item['description']}")
                if "url" in item:
                    entry.append(f"URL: {item['url']}")
                if "category" in item:
                    entry.append(f"Category: {item['category']}")
                results.append("\n".join(entry))

            return "\n---\n".join(results) if results else str(data)

        except requests.RequestException as e:
            return f"Error searching Agent101: {str(e)}"


class Agent101RecommendTool(BaseTool):
    """
    Agent101RecommendTool - Get tool recommendations from Agent101 for a given task.

    Agent101 (https://agent101.ventify.ai) is a directory/yellowpages for AI agents.
    This tool recommends the best tools from the directory for a described task.

    No API key required.
    """

    name: str = "Agent101 Recommend"
    description: str = (
        "Get AI agent tool recommendations from Agent101 (agent101.ventify.ai) "
        "for a specific task. Describe what you need to accomplish and get back "
        "recommended tools and services."
    )
    args_schema: Type[BaseModel] = Agent101RecommendSchema
    base_url: str = "https://agent101.ventify.ai/api"

    def _run(self, **kwargs: Any) -> Any:
        task = kwargs.get("task")
        if not task:
            return "Error: task description is required."

        try:
            response = requests.get(
                f"{self.base_url}/recommend",
                params={"task": task},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return f"No recommendations found for task: '{task}'."

            results = []
            for item in data:
                entry = []
                if "name" in item:
                    entry.append(f"Name: {item['name']}")
                if "description" in item:
                    entry.append(f"Description: {item['description']}")
                if "url" in item:
                    entry.append(f"URL: {item['url']}")
                if "reason" in item:
                    entry.append(f"Reason: {item['reason']}")
                results.append("\n".join(entry))

            return "\n---\n".join(results) if results else str(data)

        except requests.RequestException as e:
            return f"Error getting recommendations from Agent101: {str(e)}"
