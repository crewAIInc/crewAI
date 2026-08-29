from __future__ import annotations

from typing import Any

from pydantic import AnyHttpUrl, BaseModel, Field

from crewai_tools.tools.context_dev_tools.base import ContextDevBaseTool, compact


class ContextScrapeToolSchema(BaseModel):
    url: AnyHttpUrl = Field(
        description="Full HTTP(S) URL to scrape into clean Markdown.",
    )
    include_links: bool = Field(
        default=True,
        description="Preserve hyperlinks in the Markdown output.",
    )
    include_images: bool = Field(
        default=False,
        description="Include image references in the Markdown output.",
    )
    use_main_content_only: bool = Field(
        default=False,
        description="Remove navigation, headers, footers, and sidebars.",
    )
    include_html: bool = Field(
        default=False,
        description="Also return the source HTML used to produce the Markdown.",
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
    max_age_ms: int | None = Field(
        default=None,
        ge=0,
        le=2592000000,
        description="Maximum age of a cached result in milliseconds; use 0 for fresh data.",
    )
    wait_for_ms: int | None = Field(
        default=None,
        ge=0,
        le=30000,
        description="Extra wait after page load for JavaScript-rendered content.",
    )
    country: str | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Two-letter ISO country code for residential proxy location.",
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        le=300000,
        description="Maximum server processing time in milliseconds.",
    )


class ContextScrapeTool(ContextDevBaseTool):
    """Scrape a known web page into Markdown through Context.dev."""

    name: str = "Context.dev page scraper"
    description: str = (
        "Turn one known web page into clean, LLM-ready Markdown with Context.dev. "
        "Use search first when the exact URL is unknown."
    )
    args_schema: type[BaseModel] = ContextScrapeToolSchema

    def _run(
        self,
        url: str,
        include_links: bool = True,
        include_images: bool = False,
        use_main_content_only: bool = False,
        include_html: bool = False,
        include_selectors: list[str] | None = None,
        exclude_selectors: list[str] | None = None,
        max_age_ms: int | None = None,
        wait_for_ms: int | None = None,
        country: str | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        return self._request(
            "GET",
            "/web/scrape/markdown",
            params=compact(
                {
                    "url": url,
                    "includeLinks": include_links,
                    "includeImages": include_images,
                    "useMainContentOnly": use_main_content_only,
                    "includeHTML": include_html,
                    "includeSelectors": include_selectors,
                    "excludeSelectors": exclude_selectors,
                    "maxAgeMs": max_age_ms,
                    "waitForMs": wait_for_ms,
                    "country": country,
                    "timeoutMS": timeout_ms,
                }
            ),
        )
