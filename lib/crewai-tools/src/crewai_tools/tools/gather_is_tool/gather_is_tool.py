"""gather.is tools for CrewAI agents.

gather.is is a social network for AI agents. These tools let CrewAI agents
browse the public feed, discover other agents, and search posts.

No authentication or API keys required — all public endpoints are open.
"""

from typing import Any, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class GatherIsFeedSchema(BaseModel):
    """Input schema for GatherIsFeedTool."""

    sort: str = Field(
        "newest",
        description="Sort order: 'newest' for most recent or 'score' for highest scored",
    )
    limit: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of posts to retrieve (1-50)",
    )


class GatherIsFeedTool(BaseTool):
    """Browse the gather.is public feed — a social network for AI agents.

    Returns recent posts with titles, summaries, authors, scores, and tags.
    No API key or authentication required.
    """

    name: str = "Browse gather.is feed"
    description: str = (
        "Browse the gather.is public feed to see what AI agents are posting "
        "and discussing. Returns posts with title, summary, author, score, "
        "and tags. No authentication required."
    )
    args_schema: Type[BaseModel] = GatherIsFeedSchema
    base_url: str = "https://gather.is"

    def _run(self, **kwargs: Any) -> str:
        sort = kwargs.get("sort", "newest")
        limit = kwargs.get("limit", 10)

        try:
            response = requests.get(
                f"{self.base_url}/api/posts",
                params={"sort": sort, "limit": min(limit, 50)},
                timeout=15,
            )
            response.raise_for_status()
            posts = response.json().get("posts", [])
        except requests.RequestException as e:
            return f"Error fetching gather.is feed: {e}"

        if not posts:
            return "The gather.is feed is currently empty."

        lines = [f"Found {len(posts)} posts on gather.is:\n"]
        for post in posts:
            title = post.get("title", "Untitled")
            author = post.get("author", "unknown")
            summary = post.get("summary", "")
            score = post.get("score", 0)
            tags = ", ".join(post.get("tags", []))
            lines.append(
                f"- \"{title}\" by {author} (score: {score})"
                f"{f' [{tags}]' if tags else ''}"
                f"\n  {summary}"
            )
        return "\n".join(lines)


class GatherIsAgentsSchema(BaseModel):
    """Input schema for GatherIsAgentsTool."""

    limit: int = Field(
        20,
        ge=1,
        le=50,
        description="Number of agents to retrieve (1-50)",
    )


class GatherIsAgentsTool(BaseTool):
    """Discover agents registered on gather.is.

    Returns agent names, verification status, and post counts.
    No API key or authentication required.
    """

    name: str = "Discover gather.is agents"
    description: str = (
        "Discover AI agents registered on gather.is. Returns agent names, "
        "verification status, and post counts. No authentication required."
    )
    args_schema: Type[BaseModel] = GatherIsAgentsSchema
    base_url: str = "https://gather.is"

    def _run(self, **kwargs: Any) -> str:
        limit = kwargs.get("limit", 20)

        try:
            response = requests.get(
                f"{self.base_url}/api/agents",
                params={"limit": min(limit, 50)},
                timeout=15,
            )
            response.raise_for_status()
            agents = response.json().get("agents", [])
        except requests.RequestException as e:
            return f"Error fetching agents: {e}"

        if not agents:
            return "No agents registered on gather.is yet."

        lines = [f"Found {len(agents)} agents on gather.is:\n"]
        for agent in agents:
            name = agent.get("name", "unknown")
            verified = "verified" if agent.get("verified") else "unverified"
            post_count = agent.get("post_count", 0)
            lines.append(f"- {name} ({verified}, {post_count} posts)")
        return "\n".join(lines)


class GatherIsSearchSchema(BaseModel):
    """Input schema for GatherIsSearchTool."""

    query: str = Field(
        ...,
        description="Search query to find posts on gather.is",
    )
    limit: int = Field(
        10,
        ge=1,
        le=50,
        description="Maximum number of results (1-50)",
    )


class GatherIsSearchTool(BaseTool):
    """Search posts on gather.is by keyword.

    Returns matching posts from the agent social network.
    No API key or authentication required.
    """

    name: str = "Search gather.is posts"
    description: str = (
        "Search for posts on gather.is, a social network for AI agents. "
        "Find discussions about specific topics in the agent community. "
        "No authentication required."
    )
    args_schema: Type[BaseModel] = GatherIsSearchSchema
    base_url: str = "https://gather.is"

    def _run(self, **kwargs: Any) -> str:
        query = kwargs.get("query", kwargs.get("search_query", ""))
        limit = kwargs.get("limit", 10)

        if not query:
            return "A search query is required."

        try:
            response = requests.get(
                f"{self.base_url}/api/posts",
                params={"q": query, "limit": min(limit, 50)},
                timeout=15,
            )
            response.raise_for_status()
            posts = response.json().get("posts", [])
        except requests.RequestException as e:
            return f"Error searching gather.is: {e}"

        if not posts:
            return f"No posts found matching '{query}'."

        lines = [f"Found {len(posts)} posts matching '{query}':\n"]
        for post in posts:
            title = post.get("title", "Untitled")
            author = post.get("author", "unknown")
            summary = post.get("summary", "")
            score = post.get("score", 0)
            lines.append(
                f"- \"{title}\" by {author} (score: {score})"
                f"\n  {summary}"
            )
        return "\n".join(lines)
