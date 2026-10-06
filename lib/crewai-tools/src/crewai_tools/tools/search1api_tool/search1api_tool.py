"""Search1API tools for web search, news search, and page crawling."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, model_validator
import requests


SEARCH1API_API_URL = "https://api.search1api.com"

TimeRange = Literal["day", "week", "month", "year"]


def _error_message(response: requests.Response) -> str:
    """Return the ``message``/``error`` field of an error body, or its raw text."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or "no response body"
    if isinstance(body, dict):
        return str(body.get("message") or body.get("error") or body)[:200]
    return str(body)[:200]


class Search1APIBaseTool(BaseTool):
    """Shared API key handling and request logic for the Search1API tools."""

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SEARCH1API_API_KEY"),
        description="The Search1API API key. If not provided, it is loaded from "
        "the SEARCH1API_API_KEY environment variable.",
        exclude=True,
        repr=False,
    )
    timeout: int = Field(
        default=30, description="The timeout for each API request in seconds."
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SEARCH1API_API_KEY",
                description="API key for Search1API",
                required=True,
            ),
        ]
    )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST ``payload`` to a Search1API endpoint and return the JSON body.

        Raises:
            ValueError: If no API key is configured.
            RuntimeError: If the request fails or the response is not a JSON object.
        """
        if not self.api_key:
            raise ValueError(
                "Search1API API key is required. Set the SEARCH1API_API_KEY "
                "environment variable or pass api_key. Get a key at https://app.s1.dev"
            )
        try:
            response = requests.post(
                f"{SEARCH1API_API_URL}{path}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Search1API {path} request failed: {e}") from e

        if not response.ok:
            raise RuntimeError(
                f"Search1API {path} request failed with HTTP "
                f"{response.status_code}: {_error_message(response)}"
            )
        try:
            body = response.json()
        except ValueError as e:
            raise RuntimeError(f"Search1API {path} returned invalid JSON") from e
        if not isinstance(body, dict):
            raise RuntimeError(f"Search1API {path} returned a malformed response")
        return body


class _Search1APIResultsTool(Search1APIBaseTool):
    """Common options and result handling for the /search and /news endpoints."""

    max_results: int = Field(
        default=5, ge=1, le=50, description="The maximum number of results."
    )
    crawl_results: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Fetch the full page content of the top N results "
        "(1 extra credit per crawled page). Must not exceed max_results.",
    )
    time_range: TimeRange | None = Field(
        default=None,
        description="Only include results published within this time range.",
    )
    include_sites: list[str] = Field(
        default_factory=list,
        description="Only return results from these sites (e.g. github.com).",
    )
    exclude_sites: list[str] = Field(
        default_factory=list, description="Exclude results from these sites."
    )
    max_content_length_per_result: int = Field(
        default=4000,
        ge=1,
        description="Maximum length of crawled page content kept per result, "
        "to avoid context window issues.",
    )

    @model_validator(mode="after")
    def _check_crawl_results(self) -> _Search1APIResultsTool:
        # The API only crawls pages it returned and rejects a larger value.
        if self.crawl_results > self.max_results:
            raise ValueError("crawl_results cannot be greater than max_results")
        return self

    def _search(self, path: str, query: str, extra: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "query": query,
            "max_results": self.max_results,
            "crawl_results": self.crawl_results,
            **extra,
        }
        if self.time_range:
            payload["time_range"] = self.time_range
        if self.include_sites:
            payload["include_sites"] = self.include_sites
        if self.exclude_sites:
            payload["exclude_sites"] = self.exclude_sites

        body = self._post(path, payload)
        raw_results = body.get("results")
        if not isinstance(raw_results, list) or not all(
            isinstance(r, dict) for r in raw_results
        ):
            raise RuntimeError(f"Search1API {path} returned a malformed results list")

        results: list[dict[str, str]] = []
        for r in raw_results:
            link = r.get("link")
            # A result without a usable URL can't be cited or crawled.
            if not isinstance(link, str) or not link.strip():
                continue
            result = {
                "title": str(r.get("title") or ""),
                "url": link,
                "snippet": str(r.get("snippet") or ""),
            }
            if r.get("published_date"):
                result["published_date"] = str(r["published_date"])
            if r.get("content"):
                content = str(r["content"])
                if len(content) > self.max_content_length_per_result:
                    content = content[: self.max_content_length_per_result] + "..."
                result["content"] = content
            results.append(result)
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)


class Search1APISearchToolSchema(BaseModel):
    """Input schema for Search1APISearchTool."""

    query: str = Field(..., description="The search query.")


class Search1APISearchTool(_Search1APIResultsTool):
    """Searches the web with Search1API.

    Results can come from a general web engine (Google, Bing, Baidu, ...) or from
    a platform such as Reddit, GitHub, arXiv, or YouTube, and can optionally
    include the full content of the top pages.
    """

    name: str = "Search1API Web Search"
    description: str = (
        "Searches the web with Search1API and returns a JSON object with the "
        "title, url, and snippet of each result. Use it for real-time information."
    )
    args_schema: type[BaseModel] = Search1APISearchToolSchema
    search_service: str | None = Field(
        default=None,
        description="Engine or source to search, e.g. google, bing, baidu, reddit, "
        "github, arxiv, youtube, x, wikipedia. Leave unset to let Search1API choose.",
    )
    language: str | None = Field(
        default=None, description="Preferred result language (e.g. en, zh, ja)."
    )

    def _run(self, query: str) -> str:
        extra: dict[str, Any] = {}
        if self.search_service:
            extra["search_service"] = self.search_service
        if self.language:
            extra["language"] = self.language
        return self._search("/search", query, extra)


class Search1APINewsToolSchema(BaseModel):
    """Input schema for Search1APINewsTool."""

    query: str = Field(..., description="The news search query.")


class Search1APINewsTool(_Search1APIResultsTool):
    """Searches recent news articles with Search1API."""

    name: str = "Search1API News Search"
    description: str = (
        "Searches recent news with Search1API and returns a JSON object with the "
        "title, url, snippet, and publication date of each article."
    )
    args_schema: type[BaseModel] = Search1APINewsToolSchema
    search_service: str | None = Field(
        default=None,
        description="News source to search, e.g. google, bing, duckduckgo, yahoo, "
        "hackernews. Leave unset to let Search1API choose.",
    )

    def _run(self, query: str) -> str:
        extra: dict[str, Any] = {}
        if self.search_service:
            extra["search_service"] = self.search_service
        return self._search("/news", query, extra)


class Search1APICrawlToolSchema(BaseModel):
    """Input schema for Search1APICrawlTool."""

    url: str = Field(..., description="The URL of the web page to read.")


class Search1APICrawlTool(Search1APIBaseTool):
    """Fetches a web page with Search1API and returns its content as markdown."""

    name: str = "Search1API Crawl"
    description: str = (
        "Reads a web page with Search1API and returns a JSON object with its "
        "title, url, and main content as clean markdown."
    )
    args_schema: type[BaseModel] = Search1APICrawlToolSchema
    max_content_length: int | None = Field(
        default=None,
        ge=1,
        description="Maximum length of the returned page content. None keeps "
        "the full page.",
    )

    def _run(self, url: str) -> str:
        body = self._post("/crawl", {"url": url})
        page = body.get("results")
        if not isinstance(page, dict) or not isinstance(page.get("content"), str):
            raise RuntimeError("Search1API /crawl returned a malformed response")
        content: str = page["content"]
        if (
            self.max_content_length is not None
            and len(content) > self.max_content_length
        ):
            content = content[: self.max_content_length] + "..."
        return json.dumps(
            {
                "title": str(page.get("title") or ""),
                "url": str(page.get("link") or url),
                "content": content,
            },
            ensure_ascii=False,
        )
