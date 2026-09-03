"""MrScraper reusable scraper creation tools."""

from typing import Any

from pydantic import BaseModel

from crewai_tools.tools.mrscraper.base import MrScraperBaseTool
from crewai_tools.tools.mrscraper.payloads import (
    general_payload,
    listing_payload,
    map_payload,
)
from crewai_tools.tools.mrscraper.schemas import (
    GeneralScraperInput,
    ListingScraperInput,
    MapScraperInput,
    ScrapingMode,
)


class MrScraperCreatePromptScraperTool(MrScraperBaseTool):
    """Create a reusable General AI scraper."""

    name: str = "mrscraper_create_prompt_scraper"
    description: str = (
        "Use this to create a reusable General AI scraper from a page, prompt, and optional "
        "output schema. Use extract_page_by_prompt for immediate one-page extraction intent."
    )
    args_schema: type[BaseModel] = GeneralScraperInput

    def _run(
        self,
        url: str,
        prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
        mode: ScrapingMode = "Super",
        proxy_country: str | None = None,
    ) -> str:
        """Create a reusable General AI scraper."""
        return self._client.request(
            "POST",
            "primary",
            "/api/v1/scrapers-ai",
            json_body=general_payload(
                url=url,
                prompt=prompt,
                output_schema=output_schema,
                mode=mode,
                proxy_country=proxy_country,
            ),
        )


class MrScraperCreateListingScraperTool(MrScraperBaseTool):
    """Create a reusable Listing AI scraper."""

    name: str = "mrscraper_create_listing_scraper"
    description: str = (
        "Use this to create a reusable Listing AI scraper for repeated or paginated items. "
        "Use extract_listings when the intent is immediate extraction."
    )
    args_schema: type[BaseModel] = ListingScraperInput

    def _run(
        self,
        url: str,
        prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
        max_pages: int = 1,
        proxy_country: str | None = None,
    ) -> str:
        """Create a reusable Listing AI scraper."""
        return self._client.request(
            "POST",
            "primary",
            "/api/v1/scrapers-ai",
            json_body=listing_payload(
                url=url,
                prompt=prompt,
                output_schema=output_schema,
                max_pages=max_pages,
                proxy_country=proxy_country,
            ),
        )


class MrScraperCreateWebsiteCrawlScraperTool(MrScraperBaseTool):
    """Create a reusable Map AI scraper."""

    name: str = "mrscraper_create_website_crawl_scraper"
    description: str = (
        "Use this potentially expensive operation to create a reusable Map scraper for URL "
        "discovery. Use crawl_website_urls for immediate crawl intent."
    )
    args_schema: type[BaseModel] = MapScraperInput

    def _run(
        self,
        url: str,
        max_depth: int = 2,
        max_pages: int = 50,
        limit: int = 50,
        include_patterns: str | None = None,
        exclude_patterns: str | None = None,
    ) -> str:
        """Create a reusable Map AI scraper."""
        return self._client.request(
            "POST",
            "primary",
            "/api/v1/scrapers-ai",
            json_body=map_payload(
                url=url,
                max_depth=max_depth,
                max_pages=max_pages,
                limit=limit,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
            ),
        )


__all__ = [
    "MrScraperCreateListingScraperTool",
    "MrScraperCreatePromptScraperTool",
    "MrScraperCreateWebsiteCrawlScraperTool",
]
