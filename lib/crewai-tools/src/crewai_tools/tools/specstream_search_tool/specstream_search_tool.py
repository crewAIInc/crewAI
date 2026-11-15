import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class SpecStreamSearchToolSchema(BaseModel):
    """Input schema for SpecStreamSearchTool."""

    query: str = Field(..., description="The search query, in everyday words.")
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="How many results to return (1 to 100). Defaults to 10.",
    )
    speed: Literal["default", "fast"] | None = Field(
        default=None,
        description="'fast' returns the engines' first ranked results at a lower price.",
    )


class SpecStreamSearchTool(BaseTool):
    """Searches the live web with the SpecStream Search API.

    Attributes:
        api_key: The SpecStream API key. Read from SPECSTREAM_API_KEY if not given.
        base_url: The SpecStream API address.
        max_results: Default number of results when the agent does not ask for one.
        timeout: Request timeout in seconds.
    """

    name: str = "SpecStream Web Search"
    description: str = (
        "Searches the live web with the SpecStream Search API. Returns a JSON list "
        "of results with title, url, snippet and the published date when the page "
        "states one."
    )
    args_schema: type[BaseModel] = SpecStreamSearchToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SPECSTREAM_API_KEY"),
        description="The SpecStream API key. Read from SPECSTREAM_API_KEY if not given.",
    )
    base_url: str = "https://api.specstream.ai"
    max_results: int = 10
    timeout: int = 60
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SPECSTREAM_API_KEY",
                description="API key for SpecStream",
                required=True,
            ),
        ]
    )

    def _run(
        self,
        query: str,
        max_results: int | None = None,
        speed: Literal["default", "fast"] | None = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("SPECSTREAM_API_KEY environment variable is required")

        params: dict[str, Any] = {"q": query, "num": max_results or self.max_results}
        if speed:
            params["speed"] = speed

        try:
            response = requests.get(
                f"{self.base_url}/v1/web-search",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                params=params,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"SpecStream API request failed: {exc}") from exc

        if not response.ok:
            raise RuntimeError(
                f"SpecStream API error (HTTP {response.status_code}): {response.text[:500]}"
            )

        sources = response.json().get("data", {}).get("sources") or []
        results = [
            {
                "title": source.get("title"),
                "url": source.get("url"),
                "snippet": source.get("snippet"),
                "published": source.get("published"),
            }
            for source in sources
        ]
        return json.dumps(results, indent=2)
