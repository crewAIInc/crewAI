from __future__ import annotations

import time
from typing import Any

import requests as http_requests
from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


class CrwCrawlWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL to crawl")


class CrwCrawlWebsiteTool(BaseTool):
    """Crawl websites using CRW and return content from multiple pages.

    CRW performs async BFS crawling with rate limiting, robots.txt respect,
    and sitemap support. Runs self-hosted (free) or via fastcrw.com cloud.

    Args:
        api_url: CRW server URL. Default: http://localhost:3000
        api_key: Optional API key (required for fastcrw.com cloud).
        config: Crawl configuration options.
        poll_interval: Seconds between status checks. Default: 2
        max_wait: Maximum seconds to wait for crawl completion. Default: 300

    Configuration options:
        max_depth (int): Maximum link-follow depth. Default: 2
        max_pages (int): Maximum pages to scrape. Default: 10
        formats (list[str]): Output formats per page. Default: ["markdown"]
        onlyMainContent (bool): Strip boilerplate. Default: True
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "CRW web crawl tool"
    description: str = (
        "Crawl websites using CRW and return content from multiple pages. "
        "Useful for gathering information across an entire site."
    )
    args_schema: type[BaseModel] = CrwCrawlWebsiteToolSchema
    api_url: str = "http://localhost:3000"
    api_key: str | None = None
    poll_interval: int = Field(default=2, gt=0)
    max_wait: int = Field(default=300, gt=0)
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "maxDepth": 2,
            "maxPages": 10,
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

    def _get_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _run(self, url: str) -> Any:
        headers = self._get_headers()
        base = self.api_url.rstrip("/")

        # Start crawl job
        payload = {"url": url, **self.config}
        response = http_requests.post(
            f"{base}/v1/crawl",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            raise RuntimeError(
                f"CRW crawl start failed: {result.get('error', 'Unknown error')}"
            )

        job_id = result["id"]

        # Poll for completion
        elapsed = 0
        while elapsed < self.max_wait:
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

            status_resp = http_requests.get(
                f"{base}/v1/crawl/{job_id}",
                headers=headers,
                timeout=30,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()

            if status_data["status"] == "completed":
                pages = status_data.get("data", [])
                # Concatenate markdown from all pages
                combined = []
                for page in pages:
                    source = page.get("metadata", {}).get("sourceURL", "unknown")
                    content = page.get("markdown", "")
                    if content:
                        combined.append(f"## Source: {source}\n\n{content}")
                return "\n\n---\n\n".join(combined) if combined else "No content found."

            if status_data["status"] == "failed":
                error_detail = (
                    status_data.get("error")
                    or status_data.get("message")
                    or status_data.get("reason")
                    or str(status_data)
                )
                raise RuntimeError(f"CRW crawl job failed: {error_detail}")

        raise TimeoutError(
            f"CRW crawl did not complete within {self.max_wait} seconds"
        )
