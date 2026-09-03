"""MrScraper discovery tools."""

from typing import Literal

from pydantic import BaseModel

from crewai_tools.tools.mrscraper.base import MrScraperBaseTool
from crewai_tools.tools.mrscraper.payloads import map_payload
from crewai_tools.tools.mrscraper.schemas import MapScraperInput, SearchGoogleSerpInput


class MrScraperCrawlWebsiteUrlsTool(MrScraperBaseTool):
    """Discover URLs from a starting page."""

    name: str = "mrscraper_crawl_website_urls"
    description: str = (
        "Use this potentially expensive immediate Map crawl to discover website URLs. "
        "Use the website-crawl creation tool when the intent is to create a reusable scraper."
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
        """Discover URLs from a starting page with bounded crawl controls."""
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


class MrScraperSearchGoogleSerpTool(MrScraperBaseTool):
    """Search Google synchronously through MrScraper."""

    name: str = "mrscraper_search_google_serp"
    description: str = (
        "Use this for one narrow synchronous Google search. It returns compact JSON text "
        "for JSON format or the exact upstream HTML string for HTML format."
    )
    args_schema: type[BaseModel] = SearchGoogleSerpInput

    def _run(
        self,
        query: str,
        region: str = "us",
        language: str = "en",
        page: int = 1,
        format: Literal["json", "html"] = "json",
        render_js: bool = False,
    ) -> str:
        """Run a synchronous Google search and return JSON or HTML text."""
        return self._client.request(
            "POST",
            "serp",
            "/api/google/serp/v2/sync",
            json_body={
                "query": query,
                "region": region,
                "language": language,
                "page": page,
                "format": format,
                "renderJs": render_js,
            },
            force_text=format == "html",
        )


__all__ = ["MrScraperCrawlWebsiteUrlsTool", "MrScraperSearchGoogleSerpTool"]
