from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.context_dev_tools.base import ContextDevBaseTool, compact


class ContextSitemapToolSchema(BaseModel):
    domain: str = Field(
        min_length=3,
        description="Domain whose sitemap URLs should be discovered.",
    )
    max_links: int = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Maximum number of matching URLs to return.",
    )
    search: str | None = Field(
        default=None,
        min_length=2,
        description="Natural-language topic used to rank and filter sitemap URLs.",
    )
    url_regex: str | None = Field(
        default=None,
        description="Only return URLs matching this regular expression.",
    )
    sitemap_url: str | None = Field(
        default=None,
        description="Explicit sitemap URL to crawl instead of auto-discovery.",
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        le=300000,
        description="Maximum server processing time in milliseconds.",
    )


class ContextSitemapTool(ContextDevBaseTool):
    name: str = "Context.dev sitemap discovery"
    description: str = (
        "Discover or semantically search URLs from a website's sitemap with "
        "Context.dev without fetching every page body."
    )
    args_schema: type[BaseModel] = ContextSitemapToolSchema

    def _run(
        self,
        domain: str,
        max_links: int = 10000,
        search: str | None = None,
        url_regex: str | None = None,
        sitemap_url: str | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        return self._request(
            "GET",
            "/web/scrape/sitemap",
            params=compact(
                {
                    "domain": domain,
                    "maxLinks": max_links,
                    "search": search,
                    "urlRegex": url_regex,
                    "sitemapUrl": sitemap_url,
                    "timeoutMS": timeout_ms,
                }
            ),
        )
