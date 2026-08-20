from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.context_dev_tools.base import ContextDevBaseTool, compact


class ContextCrawlToolSchema(BaseModel):
    url: str = Field(
        min_length=1,
        description="Starting HTTP(S) URL for the crawl.",
    )
    max_pages: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of pages to crawl, from 1 to 500.",
    )
    max_depth: int | None = Field(
        default=None,
        ge=0,
        description="Maximum link depth; 0 limits the crawl to the starting page.",
    )
    url_regex: str | None = Field(
        default=None,
        description="Only follow and scrape URLs matching this regular expression.",
    )
    follow_subdomains: bool = Field(
        default=False,
        description="Follow links on subdomains of the starting domain.",
    )
    include_links: bool = Field(
        default=True,
        description="Preserve hyperlinks in each page's Markdown.",
    )
    include_images: bool = Field(
        default=False,
        description="Include image references in each page's Markdown.",
    )
    use_main_content_only: bool = Field(
        default=False,
        description="Remove navigation, headers, footers, and sidebars.",
    )
    include_selectors: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Keep only HTML subtrees matching these CSS selectors.",
    )
    exclude_selectors: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Remove HTML subtrees matching these CSS selectors.",
    )
    stop_after_ms: int | None = Field(
        default=None,
        ge=10000,
        le=110000,
        description="Soft crawl time budget in milliseconds.",
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        le=300000,
        description="Maximum server processing time in milliseconds.",
    )


class ContextCrawlTool(ContextDevBaseTool):
    name: str = "Context.dev website crawler"
    description: str = (
        "Crawl linked pages from a website and return their Markdown with "
        "Context.dev. Use this for multi-page research and documentation collection."
    )
    args_schema: type[BaseModel] = ContextCrawlToolSchema

    def _run(
        self,
        url: str,
        max_pages: int = 100,
        max_depth: int | None = None,
        url_regex: str | None = None,
        follow_subdomains: bool = False,
        include_links: bool = True,
        include_images: bool = False,
        use_main_content_only: bool = False,
        include_selectors: list[str] | None = None,
        exclude_selectors: list[str] | None = None,
        stop_after_ms: int | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        return self._request(
            "POST",
            "/web/crawl",
            json_body=compact(
                {
                    "url": url,
                    "maxPages": max_pages,
                    "maxDepth": max_depth,
                    "urlRegex": url_regex,
                    "followSubdomains": follow_subdomains,
                    "includeLinks": include_links,
                    "includeImages": include_images,
                    "useMainContentOnly": use_main_content_only,
                    "includeSelectors": include_selectors,
                    "excludeSelectors": exclude_selectors,
                    "stopAfterMs": stop_after_ms,
                    "timeoutMS": timeout_ms,
                }
            ),
        )
