"""CrewAI tools for the AlterLab web scraping API."""

from __future__ import annotations

from collections.abc import Callable
import os
import time
from typing import Any

from crewai.tools import BaseTool, EnvVar
import httpx
from pydantic import BaseModel, Field


_JOB_DONE = {"succeeded", "completed"}
_JOB_FAILED = {"failed", "rejected", "cancelled"}


def _page_content(page: dict[str, Any]) -> str:
    """Return the best text representation of a scrape or crawl page result."""
    content = page.get("content")
    if isinstance(content, dict):
        for key in ("markdown", "text", "html"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
        return ""
    if isinstance(content, str):
        return content
    for key in ("markdown", "text", "html"):
        value = page.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


class _AlterLabBase(BaseTool):
    """Shared HTTP handling for the AlterLab tools (https://api.alterlab.io)."""

    api_key: str | None = Field(
        default=None,
        description="AlterLab API key. Defaults to the ALTERLAB_API_KEY env var.",
    )
    base_url: str = Field(
        default="https://api.alterlab.io",
        description="AlterLab API base URL.",
    )
    timeout: float = Field(default=60.0, description="HTTP timeout in seconds.")
    poll_timeout: float = Field(
        default=300.0, description="Maximum seconds to wait for a job to finish."
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="ALTERLAB_API_KEY",
                description="API key for the AlterLab web scraping API",
                required=True,
            ),
        ]
    )

    def _http(self) -> httpx.Client:
        key = self.api_key or os.environ.get("ALTERLAB_API_KEY")
        if not key:
            raise ValueError(
                "AlterLab API key is required. Pass api_key or set ALTERLAB_API_KEY."
            )
        return httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers={"X-API-Key": key},
            timeout=self.timeout,
        )

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        if response.is_error:
            raise RuntimeError(
                f"AlterLab API error {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        return data if isinstance(data, dict) else {}

    def _wait(
        self, fetch: Callable[[], dict[str, Any]], done: set[str], failed: set[str]
    ) -> dict[str, Any]:
        """Poll ``fetch()`` until its ``status`` is terminal."""
        deadline = time.monotonic() + self.poll_timeout
        delay = 1.0
        while True:
            data = fetch()
            status = data.get("status")
            if status in done:
                return data
            if status in failed:
                raise RuntimeError(
                    f"AlterLab job {status}: {data.get('error') or 'no detail'}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError("AlterLab job did not finish before poll_timeout")
            time.sleep(delay)
            delay = min(delay * 1.5, 10.0)


class AlterLabScrapeToolSchema(BaseModel):
    url: str = Field(description="URL of the web page to scrape")


class AlterLabScrapeTool(_AlterLabBase):
    """Scrape a single web page with AlterLab and return its content.

    AlterLab handles anti-bot protection and JavaScript rendering. The lowest
    cost tier that can fetch the page is used automatically.

    Args:
        api_key: AlterLab API key. Defaults to the ``ALTERLAB_API_KEY`` env var.
        scrape_params: Extra keyword arguments passed to the ``/api/v1/scrape`` request body,
            for example ``{"mode": "js", "formats": ["markdown"]}``. LLM-based
            extraction options (``extraction_prompt`` / ``extraction_schema``)
            require your own LLM provider key registered in AlterLab (BYOK).

    Example:
        >>> tool = AlterLabScrapeTool()
        >>> tool.run(url="https://example.com")
    """

    name: str = "AlterLab web scraper"
    description: str = (
        "Scrape a web page and return its content as markdown. "
        "Works on JavaScript-heavy and bot-protected sites."
    )
    args_schema: type[BaseModel] = AlterLabScrapeToolSchema
    scrape_params: dict[str, Any] = Field(
        default_factory=lambda: {"formats": ["markdown"]}
    )

    def _run(self, url: str) -> str:
        payload = {"url": url, "mode": "auto", "sync": True, **self.scrape_params}
        with self._http() as http:
            response = http.post("/api/v1/scrape", json=payload)
            result = self._json(response)
            if response.status_code == 202 and result.get("job_id"):
                job_id = result["job_id"]
                job = self._wait(
                    lambda: self._json(http.get(f"/api/v1/jobs/{job_id}")),
                    _JOB_DONE,
                    _JOB_FAILED,
                )
                result = job.get("result") or job
        return _page_content(result) or f"No content could be extracted from {url}"


class AlterLabSearchToolSchema(BaseModel):
    query: str = Field(description="Search query")


class AlterLabSearchTool(_AlterLabBase):
    """Search the web with AlterLab and return titles, URLs and snippets.

    Args:
        api_key: AlterLab API key. Defaults to the ``ALTERLAB_API_KEY`` env var.
        num_results: Number of results to return (1-30).
        search_params: Extra keyword arguments passed to the ``/api/v1/search`` request body,
            for example ``{"country": "us", "time_range": "week"}``.

    Example:
        >>> tool = AlterLabSearchTool(num_results=5)
        >>> tool.run(query="web scraping api")
    """

    name: str = "AlterLab web search"
    description: str = "Search the web and return result titles, URLs and snippets."
    args_schema: type[BaseModel] = AlterLabSearchToolSchema
    num_results: int = Field(default=10, ge=1, le=30)
    search_params: dict[str, Any] = Field(default_factory=dict)

    def _run(self, query: str) -> str:
        payload = {
            "query": query,
            "num_results": self.num_results,
            **self.search_params,
        }
        with self._http() as http:
            response = self._json(http.post("/api/v1/search", json=payload))
        results = response.get("results") or []
        if not results:
            return f"No results found for {query}"
        lines = []
        for index, item in enumerate(results, start=1):
            lines.append(f"{index}. {item.get('title', '')}\n{item.get('url', '')}")
            snippet = item.get("snippet") or item.get("description")
            if snippet:
                lines.append(str(snippet))
        return "\n".join(lines)


class AlterLabCrawlToolSchema(BaseModel):
    url: str = Field(description="Starting URL for the crawl")


class AlterLabCrawlTool(_AlterLabBase):
    """Crawl a website with AlterLab and return the content of each page.

    Args:
        api_key: AlterLab API key. Defaults to the ``ALTERLAB_API_KEY`` env var.
        max_pages: Maximum number of pages to crawl.
        max_depth: Maximum link depth from the starting URL.
        crawl_params: Extra keyword arguments passed to
            the ``/api/v1/crawl`` request body, for example
            ``{"include_patterns": ["/docs/*"]}``.

    Example:
        >>> tool = AlterLabCrawlTool(max_pages=10)
        >>> tool.run(url="https://docs.example.com")
    """

    name: str = "AlterLab website crawler"
    description: str = (
        "Crawl a website starting from a URL and return the markdown content "
        "of each page found."
    )
    args_schema: type[BaseModel] = AlterLabCrawlToolSchema
    max_pages: int = Field(default=10, ge=1)
    max_depth: int = Field(default=2, ge=0)
    crawl_params: dict[str, Any] = Field(default_factory=dict)

    def _run(self, url: str) -> str:
        payload = {
            "url": url,
            "max_pages": self.max_pages,
            "max_depth": self.max_depth,
            "formats": ["markdown"],
            **self.crawl_params,
        }
        with self._http() as http:
            crawl_id = self._json(http.post("/api/v1/crawl", json=payload))["crawl_id"]
            result = self._wait(
                lambda: self._json(
                    http.get(
                        f"/api/v1/crawl/{crawl_id}",
                        params={"include_results": "true"},
                    )
                ),
                {"completed", "partial"},
                {"failed"},
            )
        parts = []
        for page in result.get("pages") or []:
            content = _page_content(page)
            if content:
                parts.append(f"## {page.get('url', '')}\n\n{content}")
        return "\n\n---\n\n".join(parts) or f"No content could be crawled from {url}"
