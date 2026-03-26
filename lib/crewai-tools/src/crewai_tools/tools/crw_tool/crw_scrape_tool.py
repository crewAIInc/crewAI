from __future__ import annotations

from typing import Any

import requests as http_requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


class CrwScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL to scrape")


class CrwScrapeWebsiteTool(BaseTool):
    """Scrape webpages using CRW and return clean markdown content.

    CRW is an open-source web scraper built for AI agents. It can run
    self-hosted (free) or via the managed cloud at fastcrw.com.

    Args:
        api_url: CRW server URL. Default: http://localhost:3000
        api_key: Optional API key (required for fastcrw.com cloud).
        config: Scrape configuration options.

    Configuration options:
        formats (list[str]): Output formats. Default: ["markdown"]
            Options: "markdown", "html", "rawHtml", "plainText", "links", "json"
        onlyMainContent (bool): Strip nav/footer/sidebar. Default: True
        renderJs (bool|None): None=auto, True=force JS, False=HTTP only. Default: None
        wait_for (int): ms to wait after JS rendering. Default: None
        include_tags (list[str]): CSS selectors to include. Default: []
        exclude_tags (list[str]): CSS selectors to exclude. Default: []
        css_selector (str): Extract specific CSS selector. Default: None
        xpath (str): Extract specific XPath. Default: None
        headers (dict): Custom HTTP headers. Default: {}
        json_schema (dict): JSON Schema for LLM extraction. Default: None
        proxy (str): Per-request proxy URL. Default: None
        stealth (bool): Enable stealth mode. Default: None
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "CRW web scrape tool"
    description: str = (
        "Scrape webpages using CRW and return clean markdown content. "
        "CRW is a fast, open-source web scraper with JS rendering support."
    )
    args_schema: type[BaseModel] = CrwScrapeWebsiteToolSchema
    api_url: str = "http://localhost:3000"
    api_key: str | None = None
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "formats": ["markdown"],
            "onlyMainContent": True,
        }
    )

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="CRW_API_URL",
                description="CRW server URL (default: http://localhost:3000)",
                required=False,
                default="http://localhost:3000",
            ),
            EnvVar(
                name="CRW_API_KEY",
                description="API key for CRW (required for fastcrw.com cloud)",
                required=False,
            ),
        ]
    )

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        import os

        self.api_url = api_url or os.getenv("CRW_API_URL", "http://localhost:3000")
        self.api_key = api_key or os.getenv("CRW_API_KEY")

    def _run(self, url: str) -> Any:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {"url": url, **self.config}

        response = http_requests.post(
            f"{self.api_url.rstrip('/')}/v1/scrape",
            json=payload,
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            raise RuntimeError(
                f"CRW scrape failed: {result.get('error', 'Unknown error')}"
            )

        data = result["data"]

        # Return markdown by default, fall back to other formats
        if data.get("markdown"):
            return data["markdown"]
        if data.get("plainText"):
            return data["plainText"]
        if data.get("html"):
            return data["html"]
        if data.get("json"):
            return data["json"]
        return data
