from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.context_dev_tools.base import ContextDevBaseTool, compact


class ContextSearchToolSchema(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="Web search query, including optional search operators.",
    )
    num_results: int = Field(
        default=10,
        ge=10,
        le=100,
        description="Number of ranked results to return, from 10 to 100.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Only return results from these domains.",
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="Exclude results from these domains.",
    )
    freshness: (
        Literal["last_24_hours", "last_week", "last_month", "last_year"] | None
    ) = Field(
        default=None,
        description="Optional recency window for search results.",
    )
    country: str | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Two-letter ISO country code used to localize results.",
    )
    query_fanout: bool | None = Field(
        default=None,
        description="Expand the query into parallel variants for broader recall.",
    )
    include_markdown: bool = Field(
        default=False,
        description="Scrape each ranked result and include its Markdown content.",
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        le=300000,
        description="Maximum server processing time in milliseconds.",
    )


class ContextSearchTool(ContextDevBaseTool):
    """Search the live web through Context.dev."""

    name: str = "Context.dev web search"
    description: str = (
        "Search the live web with Context.dev. Use this when the exact source URL "
        "is unknown or when current, cited sources are needed."
    )
    args_schema: type[BaseModel] = ContextSearchToolSchema

    def _run(
        self,
        query: str,
        num_results: int = 10,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        freshness: Literal["last_24_hours", "last_week", "last_month", "last_year"]
        | None = None,
        country: str | None = None,
        query_fanout: bool | None = None,
        include_markdown: bool = False,
        timeout_ms: int | None = None,
    ) -> Any:
        return self._request(
            "POST",
            "/web/search",
            json_body=compact(
                {
                    "query": query,
                    "numResults": num_results,
                    "includeDomains": include_domains,
                    "excludeDomains": exclude_domains,
                    "freshness": freshness,
                    "country": country,
                    "queryFanout": query_fanout,
                    "markdownOptions": (
                        {"enabled": True, "useMainContentOnly": True}
                        if include_markdown
                        else None
                    ),
                    "timeoutMS": timeout_ms,
                }
            ),
        )
