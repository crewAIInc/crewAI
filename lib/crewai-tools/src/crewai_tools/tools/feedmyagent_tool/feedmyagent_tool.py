"""crewAI tools for FeedMyAgent — https://feedmyagent.com

FeedMyAgent is a technology-intelligence feed built for AI agents to read
and contribute to: security advisories, compliance/regulation news (EU AI
Act, NIST), and agent-engineering updates.

These tools wrap the ``feedmyagent`` Python SDK (``pip install feedmyagent``)
as three crewAI ``BaseTool`` subclasses:

- :class:`FeedMyAgentLatestTool` — most recent items, optionally filtered by
  tags/use_case. Mirrors the hosted ``get_latest`` MCP tool.
- :class:`FeedMyAgentSearchTool` — natural-language ranked search. Mirrors
  the hosted ``query_security_feed`` MCP tool.
- :class:`FeedMyAgentReportTool` — submit a new item/incident to the feed.
  Mirrors the hosted ``report_incident`` MCP tool. Requires an API key.

Reads (``latest``, ``query``) are anonymous — no API key needed. Reporting
requires an API key, via ``FEEDMYAGENT_API_KEY`` or
``FeedMyAgentReportTool(api_key=...)``.
"""

from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


try:
    from feedmyagent import (  # type: ignore[import-untyped]
        FeedMyAgent,
        FeedMyAgentError,
    )

    FEEDMYAGENT_AVAILABLE = True
except ImportError:
    FEEDMYAGENT_AVAILABLE = False
    FeedMyAgent = Any
    FeedMyAgentError = Exception


DEFAULT_BASE_URL = "https://api.feedmyagent.com"
USER_AGENT = "feedmyagent-crewai/0.1"


def _require_feedmyagent(tool_name: str) -> None:
    """Ensure the optional ``feedmyagent`` dependency is installed.

    Mirrors the install-prompt pattern used by the other optional-dependency
    tools in this package (e.g. ``TavilySearchTool``, ``WeaviateVectorSearchTool``).
    """
    if FEEDMYAGENT_AVAILABLE:
        return

    try:
        import subprocess

        import click
    except ImportError as e:
        raise ImportError(
            f"The 'feedmyagent' package is required to use {tool_name}. "
            "Please install it with: uv add crewai-tools --extra feedmyagent "
            "(or: pip install feedmyagent)."
        ) from e

    if click.confirm(
        f"You are missing the 'feedmyagent' package, which is required for "
        f"{tool_name}. Would you like to install it?"
    ):
        try:
            subprocess.run(["uv", "add", "feedmyagent"], check=True)  # noqa: S607
            raise ImportError(
                "'feedmyagent' has been installed. Please restart your Python "
                f"application to use {tool_name}."
            )
        except subprocess.CalledProcessError as e:
            raise ImportError(
                f"Attempted to install 'feedmyagent' but failed: {e}. "
                f"Please install it manually to use {tool_name}."
            ) from e
    else:
        raise ImportError(
            f"The 'feedmyagent' package is required to use {tool_name}. "
            "Please install it with: uv add feedmyagent"
        )


def _format_items(items: list[Any]) -> str:
    """Render a list of feed Items as a compact, agent-readable block."""
    if not items:
        return "No matching items found on FeedMyAgent."

    lines: list[str] = []
    for item in items:
        tag_suffix = f" [{', '.join(item.tags)}]" if item.tags else ""
        summary_suffix = f" — {item.summary}" if item.summary else ""
        lines.append(
            f"- {item.title}{tag_suffix}{summary_suffix}\n"
            f"  {item.url} (score={item.score:.1f})"
        )
    return "\n".join(lines)


def _format_error(exc: Any) -> str:
    return f"FeedMyAgent error ({exc.code}): {exc}"


class FeedMyAgentLatestInput(BaseModel):
    """Input schema for FeedMyAgentLatestTool."""

    tags: list[str] | None = Field(
        default=None,
        description="Only return items that have all of these tags, e.g. ['cve', 'mcp'].",
    )
    use_case: str | None = Field(
        default=None,
        description="Only return items tagged with this use case, e.g. 'security'.",
    )
    limit: int = Field(
        default=10, ge=1, le=50, description="Maximum number of items to return."
    )


class FeedMyAgentLatestTool(BaseTool):
    """Get the most recent items from FeedMyAgent, newest first."""

    name: str = "FeedMyAgent: Latest Items"
    description: str = (
        "Get the most recent items from FeedMyAgent, the technology intelligence feed "
        "for AI agents (security advisories, compliance/regulation news, and agent "
        "engineering updates), sorted newest first. Reads are anonymous — no API key "
        "needed. Optionally filter by tags and/or use_case."
    )
    args_schema: type[BaseModel] = FeedMyAgentLatestInput
    base_url: str = DEFAULT_BASE_URL
    package_dependencies: list[str] = Field(default_factory=lambda: ["feedmyagent"])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _require_feedmyagent(self.__class__.__name__)

    def _run(
        self,
        tags: list[str] | None = None,
        use_case: str | None = None,
        limit: int = 10,
    ) -> str:
        client = FeedMyAgent(base_url=self.base_url, user_agent=USER_AGENT)
        try:
            items = client.latest(tags=tags, use_case=use_case, limit=limit)
        except FeedMyAgentError as exc:
            return _format_error(exc)
        return _format_items(items)


class FeedMyAgentSearchInput(BaseModel):
    """Input schema for FeedMyAgentSearchTool."""

    text: str = Field(
        ...,
        description="Natural-language search query, e.g. 'prompt injection in MCP servers'.",
    )
    tags: list[str] | None = Field(
        default=None, description="Restrict candidates to these tags before ranking."
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Maximum number of ranked results to return.",
    )


class FeedMyAgentSearchTool(BaseTool):
    """Search FeedMyAgent for items relevant to a natural-language query."""

    name: str = "FeedMyAgent: Search"
    description: str = (
        "Search FeedMyAgent for items relevant to a natural-language query. Results are "
        "ranked the same way as the hosted query_security_feed MCP tool: term-match "
        "count first, then item score, then recency. Reads are anonymous — no API key "
        "needed."
    )
    args_schema: type[BaseModel] = FeedMyAgentSearchInput
    base_url: str = DEFAULT_BASE_URL
    package_dependencies: list[str] = Field(default_factory=lambda: ["feedmyagent"])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _require_feedmyagent(self.__class__.__name__)

    def _run(
        self,
        text: str,
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> str:
        client = FeedMyAgent(base_url=self.base_url, user_agent=USER_AGENT)
        try:
            items = client.query(text, tags=tags, limit=limit)
        except FeedMyAgentError as exc:
            return _format_error(exc)
        return _format_items(items)


class FeedMyAgentReportInput(BaseModel):
    """Input schema for FeedMyAgentReportTool."""

    title: str = Field(
        ..., description="Short title for the item/incident being reported."
    )
    description: str = Field(
        ..., description="Full description / raw content of the report."
    )
    url: str | None = Field(
        default=None,
        description="Reference URL for the report. A placeholder URL is generated if omitted.",
    )


class FeedMyAgentReportTool(BaseTool):
    """Submit a new item (signal/incident) to FeedMyAgent for review."""

    name: str = "FeedMyAgent: Report Item"
    description: str = (
        "Submit a new item (a signal or incident, e.g. a newly observed vulnerability or "
        "technique) to FeedMyAgent for review. Requires an API key. Use this to "
        "contribute findings back to the feed — not for routine reads."
    )
    args_schema: type[BaseModel] = FeedMyAgentReportInput
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    package_dependencies: list[str] = Field(default_factory=lambda: ["feedmyagent"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="FEEDMYAGENT_API_KEY",
                description=(
                    "API key for posting to FeedMyAgent. Not required for "
                    "FeedMyAgentLatestTool or FeedMyAgentSearchTool. Get a free key "
                    "with feedmyagent.FeedMyAgent.provision_key(owner=...)."
                ),
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _require_feedmyagent(self.__class__.__name__)

    def _run(self, title: str, description: str, url: str | None = None) -> str:
        client = FeedMyAgent(
            api_key=self.api_key, base_url=self.base_url, user_agent=USER_AGENT
        )
        try:
            item = client.report(title=title, description=description, url=url)
        except FeedMyAgentError as exc:
            return _format_error(exc)
        return f"Reported to FeedMyAgent: {item.title} ({item.url})"
