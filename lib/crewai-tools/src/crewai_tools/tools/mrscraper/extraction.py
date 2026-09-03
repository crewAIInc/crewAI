"""MrScraper immediate extraction tools."""

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from crewai_tools.tools.mrscraper.base import MrScraperBaseTool
from crewai_tools.tools.mrscraper.payloads import (
    general_payload,
    listing_payload,
    rendered_request,
)
from crewai_tools.tools.mrscraper.schemas import (
    ExtractStructuredDataInput,
    FetchRenderedHtmlInput,
    GeneralScraperInput,
    ListingScraperInput,
    ScrapingMode,
    StructuredDataCategory,
)


@lru_cache(maxsize=1)
def load_structured_data_prompts() -> dict[str, str]:
    """Load the byte-preserved n8n structured extraction presets."""
    path = Path(__file__).with_name("structured_data_prompts.json")
    with path.open(encoding="utf-8") as preset_file:
        return cast(dict[str, str], json.load(preset_file))


class MrScraperExtractPageByPromptTool(MrScraperBaseTool):
    """Perform immediate General extraction."""

    name: str = "mrscraper_extract_page_by_prompt"
    description: str = (
        "Use this for immediate AI extraction from one page using a prompt. "
        "Use create_prompt_scraper when the primary intent is reusable scraper creation."
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
        """Extract prompted data from one page immediately."""
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


class MrScraperExtractListingsTool(MrScraperBaseTool):
    """Perform immediate Listing extraction."""

    name: str = "mrscraper_extract_listings"
    description: str = (
        "Use this potentially multi-page immediate extraction for repeated listings or "
        "paginated content. Use create_listing_scraper for reusable scraper creation."
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
        """Extract repeated listings across one or more pages immediately."""
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


class MrScraperExtractStructuredDataTool(MrScraperBaseTool):
    """Extract one of the bundled structured-data presets."""

    name: str = "mrscraper_extract_structured_data"
    description: str = (
        "Use this immediate extraction tool when a page matches one supported structured "
        "category, such as article, product, hotel, job, property, restaurant, or tour."
    )
    args_schema: type[BaseModel] = ExtractStructuredDataInput

    def _run(
        self,
        url: str,
        category: StructuredDataCategory = "article",
        mode: ScrapingMode = "Super",
        proxy_country: str | None = None,
    ) -> str:
        """Extract data using the selected bundled structured-data preset."""
        payload: dict[str, Any] = {
            "graph": "general",
            "url": url,
            "message": load_structured_data_prompts()[category],
            "mode": mode,
        }
        if proxy_country is not None:
            payload["proxyCountry"] = proxy_country
        return self._client.request(
            "POST", "primary", "/api/v1/scrapers-ai", json_body=payload
        )


class MrScraperFetchRenderedHtmlTool(MrScraperBaseTool):
    """Fetch a browser-rendered page."""

    name: str = "mrscraper_fetch_rendered_html"
    description: str = (
        "Use this immediate stealth-browser call when JavaScript-rendered HTML, Markdown, "
        "cookies, or a screenshot is needed. Keep the requested outputs narrow to control cost."
    )
    args_schema: type[BaseModel] = FetchRenderedHtmlInput

    def _run(
        self,
        url: str,
        max_retries: int = 3,
        timeout: int = 300,
        geo_code: str = "us",
        proxy_country: str = "us",
        screenshot: bool = False,
        screenshot_mode: Literal["full", "top"] | None = None,
        html: bool = True,
        markdown: bool = False,
        token_cap: int | None = None,
        wait_for_selector: str | None = None,
        wait_until: Literal["domcontentloaded", "load", "networkidle"] | None = None,
        block_resources: bool = False,
        home_page: bool = False,
        return_cookie: bool = False,
        super_mode: bool = False,
    ) -> str:
        """Fetch a rendered page with optional browser outputs and controls."""
        params, body = rendered_request(
            url=url,
            max_retries=max_retries,
            timeout=timeout,
            geo_code=geo_code,
            proxy_country=proxy_country,
            screenshot=screenshot,
            screenshot_mode=screenshot_mode,
            html=html,
            markdown=markdown,
            token_cap=token_cap,
            wait_for_selector=wait_for_selector,
            wait_until=wait_until,
            block_resources=block_resources,
            home_page=home_page,
            return_cookie=return_cookie,
            super_mode=super_mode,
        )
        return self._client.request(
            "POST", "rendered", "/", params=params, json_body=body
        )


__all__ = [
    "MrScraperExtractListingsTool",
    "MrScraperExtractPageByPromptTool",
    "MrScraperExtractStructuredDataTool",
    "MrScraperFetchRenderedHtmlTool",
    "load_structured_data_prompts",
]
