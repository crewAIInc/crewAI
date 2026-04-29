from __future__ import annotations

import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class PerplexitySearchInput(BaseModel):
    """Input schema for PerplexitySearchTool using Perplexity's Search API."""

    query: str = Field(
        ...,
        description="The search query to send to Perplexity's Search API.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        description="Maximum number of search results to return.",
    )
    search_domain_filter: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of domains to allow or deny. Prefix a domain with '-' "
            "to exclude it (e.g. '-pinterest.com'). Do not mix allow and deny in "
            "the same list."
        ),
    )
    search_recency_filter: Literal["hour", "day", "week", "month", "year"] | None = (
        Field(
            default=None,
            description="Restrict results to a recency window.",
        )
    )


class PerplexitySearchTool(BaseTool):
    """Search the web with Perplexity's Search API.

    Returns ranked web results (title, url, snippet, date) suitable for
    grounding LLM answers.
    """

    name: str = "Perplexity Search"
    description: str = (
        "Search the web using Perplexity's Search API. Returns a ranked list of "
        "web results (title, URL, snippet, date) for a given query. Supports "
        "optional domain and recency filters."
    )
    args_schema: type[BaseModel] = PerplexitySearchInput

    api_key: str | None = Field(
        default=None,
        description="API key for Perplexity. Falls back to PERPLEXITY_API_KEY / PPLX_API_KEY.",
    )
    search_url: str = "https://api.perplexity.ai/search"

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="PERPLEXITY_API_KEY",
                description="API key for Perplexity",
                required=True,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])

    def _resolve_api_key(self) -> str | None:
        return (
            self.api_key
            or os.environ.get("PERPLEXITY_API_KEY")
            or os.environ.get("PPLX_API_KEY")
        )

    def _run(
        self,
        query: str,
        max_results: int = 5,
        search_domain_filter: list[str] | None = None,
        search_recency_filter: str | None = None,
        **_: Any,
    ) -> str:
        api_key = self._resolve_api_key()
        if not api_key:
            return (
                "Error: PERPLEXITY_API_KEY (or PPLX_API_KEY) environment variable "
                "is required."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "query": query,
            "max_results": max_results,
        }
        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter

        try:
            resp = requests.post(
                self.search_url, json=payload, headers=headers, timeout=30
            )
        except requests.Timeout:
            return "Perplexity Search API timeout. Please try again later."
        except requests.RequestException as exc:
            return f"Unexpected error calling Perplexity Search API: {exc}"

        if resp.status_code >= 300:
            return f"Perplexity Search API error: {resp.status_code} {resp.text[:200]}"

        try:
            data = resp.json()
        except ValueError:
            return (
                f"Perplexity Search API returned non-JSON response: {resp.text[:200]}"
            )

        return self._format_results(data)

    @staticmethod
    def _format_results(data: dict[str, Any]) -> str:
        results = data.get("results") or []
        if not results:
            return json.dumps(data or {}, ensure_ascii=False)

        lines: list[str] = []
        for idx, item in enumerate(results, start=1):
            title = item.get("title") or "(no title)"
            url = item.get("url") or ""
            snippet = item.get("snippet") or ""
            date = item.get("date")
            header = f"{idx}. {title}\n   {url}"
            if date:
                header += f"\n   Date: {date}"
            if snippet:
                header += f"\n   {snippet}"
            lines.append(header)
        return "\n\n".join(lines)
