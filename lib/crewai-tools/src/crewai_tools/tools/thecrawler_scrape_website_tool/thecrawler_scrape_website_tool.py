from __future__ import annotations

import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field
import requests

from crewai_tools.security.safe_path import validate_url


THECRAWLER_API_BASE = "https://www.miaibot.ai/api/v1"


class TheCrawlerScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL to scrape")


class TheCrawlerScrapeWebsiteTool(BaseTool):
    """Tool for scraping webpages via the TheCrawler hosted REST API.

    TheCrawler returns boilerplate-stripped Markdown suitable for RAG ingestion,
    along with structured metadata (title, description, links, JSON-LD, OG/Twitter
    tags, commerce data, forms, analytics-tracker detection). On failure it
    returns a structured ``errorType`` enum (``dns``, ``timeout``, ``rate-limit``,
    ``blocked-bot``, ``js-required``, ``http-4xx``, ``http-5xx``, ``parse``,
    ``network``, ``unknown``) and an ``errorRetryable`` flag so agents can
    branch on the failure mode instead of regex-matching error strings.

    The hosted endpoint at ``https://www.miaibot.ai/api/v1/crawl`` is a
    pay-per-use service authenticated with a Bearer API key. The underlying
    scraper is open source (AGPL-3.0); using the hosted API does not bind
    consumer code to AGPL-3.0.

    Args:
        api_key: TheCrawler API key (``mai_live_<32-hex>``). Defaults to the
            ``THECRAWLER_API_KEY`` environment variable.
        config: Optional dict of extra request-body fields forwarded to
            ``POST /v1/crawl``. See "Default configuration" below.
        api_base: Override the API base URL. Defaults to
            ``https://www.miaibot.ai/api/v1``.
        timeout: HTTP request timeout in seconds. Defaults to 60.

    Default configuration (forwarded to the crawl endpoint):
        extractMarkdown (bool): Return boilerplate-stripped Markdown. Default: True
        usePlaywright (bool): Force Playwright rendering for JS-heavy pages.
            Default: False (engine adaptively falls back to Playwright on its own).
        retries (int): Retries on transient failures (5xx, network, timeout).
            Default: 3.
        timeout (int): Per-request fetch timeout in ms. Default: 30000.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "TheCrawler web scrape tool"
    description: str = (
        "Scrape a webpage using TheCrawler and return boilerplate-stripped "
        "Markdown plus structured metadata. Useful for RAG ingestion and "
        "agentic web research."
    )
    args_schema: type[BaseModel] = TheCrawlerScrapeWebsiteToolSchema
    api_key: str | None = None
    api_base: str = THECRAWLER_API_BASE
    timeout: int = 60
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "extractMarkdown": True,
        }
    )

    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="THECRAWLER_API_KEY",
                description="API key for TheCrawler hosted API (miaibot.ai).",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        api_key: str | None = None,
        config: dict[str, Any] | None = None,
        api_base: str | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("THECRAWLER_API_KEY")
        if config is not None:
            # Merge into defaults so caller can override individual fields
            # without dropping the rest (e.g. retries, timeoutSecs).
            self.config = {**self.config, **config}
        if api_base is not None:
            self.api_base = api_base
        if timeout is not None:
            self.timeout = timeout

    def _run(self, url: str) -> Any:
        if not self.api_key:
            raise ValueError(
                "TheCrawler API key is required. Pass api_key=... or set "
                "THECRAWLER_API_KEY in the environment."
            )

        url = validate_url(url)
        # `urls` is set last so it cannot be overridden by `self.config`.
        payload: dict[str, Any] = {**self.config, "urls": [url]}

        response = requests.post(
            f"{self.api_base}/crawl",
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
