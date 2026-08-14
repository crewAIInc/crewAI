import json
import os
from typing import Any
from urllib.parse import urlsplit

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


DEFAULT_BASE_URL = "https://api.keenable.ai"


class KeenableSearchToolSchema(BaseModel):
    """Input for KeenableSearchTool."""

    query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )


class KeenableSearchTool(BaseTool):
    """
    KeenableSearchTool - a web search tool powered by the Keenable Search API.

    Keenable is a web search API built for AI agents. Unlike most search tools,
    it works without an API key by default: with no ``KEENABLE_API_KEY`` set it
    uses the keyless public endpoint, so the tool works out of the box. Setting
    ``KEENABLE_API_KEY`` uses the authenticated endpoint and lifts rate limits.

    Dependencies:
        - requests
        - pydantic
        - python-dotenv (optional, for API key management)
    """

    name: str = "Keenable Web Search"
    description: str = (
        "A tool that searches the internet with a search query, powered by "
        "Keenable. Returns a JSON list of results (title, url, description)."
    )
    args_schema: type[BaseModel] = KeenableSearchToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("KEENABLE_API_KEY"),
        description="Keenable API key. Optional — without it the keyless free tier is used.",
    )
    base_url: str = Field(
        default_factory=lambda: os.getenv("KEENABLE_API_URL") or DEFAULT_BASE_URL,
        description="Keenable API base URL (HTTPS). Defaults to https://api.keenable.ai.",
    )
    mode: str = Field(
        default="pro",
        description="Search mode. 'pro' is the default and the only mode needed here.",
    )
    n_results: int = Field(
        default=10, ge=0, description="Maximum number of results to return."
    )
    max_snippet_chars: int = Field(
        default=500,
        ge=0,
        description=(
            "Truncate each result's description to this many characters. Keenable "
            "returns page text, which is far longer than a typical search snippet, "
            "so this keeps a result set from crowding the agent's context. "
            "Set to 0 for no truncation."
        ),
    )
    timeout: int = Field(default=30, description="Request timeout in seconds.")
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="KEENABLE_API_KEY",
                description=(
                    "API key for Keenable search (optional; keyless free tier by default)"
                ),
                required=False,
            ),
        ]
    )

    def _resolved_base_url(self) -> str:
        base = (self.base_url or DEFAULT_BASE_URL).rstrip("/")
        parsed = urlsplit(base)
        if parsed.hostname:
            if parsed.scheme == "https":
                return base
            if parsed.scheme == "http" and parsed.hostname in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                return base
        raise ValueError(
            f"KEENABLE_API_URL must be an https:// URL with a host, got {base!r}"
        )

    def _result_text(self, result: dict[str, Any]) -> str:
        """Return a result's text, whitespace-collapsed and length-capped.

        The API returns both "snippet" and "description": "snippet" holds the page
        text and "description" is usually empty, so reading "description" first
        would return results with no text at all. Snippets are raw page text and
        arrive with newlines, hence the collapse.
        """
        text = " ".join(
            str(result.get("snippet") or result.get("description") or "").split()
        )
        if 0 < self.max_snippet_chars < len(text):
            return text[: self.max_snippet_chars].rstrip() + "…"
        return text

    def _run(self, **kwargs: Any) -> Any:
        search_query = kwargs.get("query") or kwargs.get("search_query")
        if not search_query:
            raise ValueError("Search query is required")

        api_key = (self.api_key or "").strip()
        # Keyless public endpoint by default; keyed endpoint + X-API-Key with a key.
        path = "/v1/search" if api_key else "/v1/search/public"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "keenable-crewai",
            # Attribution header the Keenable backend segments traffic by.
            "X-Keenable-Title": "CrewAI",
        }
        if api_key:
            headers["X-API-Key"] = api_key

        try:
            response = requests.post(
                f"{self._resolved_base_url()}{path}",
                headers=headers,
                json={"query": search_query, "mode": self.mode},
                timeout=self.timeout,
            )
            response.raise_for_status()
            # response.json() raises ValueError on a non-JSON body.
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            return f"Error performing search: {e!s}"

        # Guard against a malformed response shape (not a dict / results not a list).
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return "Error performing search: unexpected response from the Keenable API."

        return json.dumps(
            [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "description": self._result_text(r),
                }
                for r in results[: self.n_results]
                if isinstance(r, dict) and r.get("url")
            ],
            indent=2,
        )
