import os
from typing import Any, TypedDict

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class KeirolabsResult(TypedDict):
    """A single Keirolabs web search result."""

    title: str
    link: str
    snippet: str


class KeirolabsSearchToolSchema(BaseModel):
    """Input for KeirolabsSearchTool."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )


class KeirolabsSearchTool(BaseTool):
    name: str = "Search the internet with Keirolabs"
    description: str = (
        "A tool that can be used to search the internet with a search_query. "
        "Returns a list of results, each with a title, link and snippet."
    )
    args_schema: type[BaseModel] = KeirolabsSearchToolSchema
    base_url: str = "https://api.keirolabs.cloud"
    n_results: int = 10
    timeout: int = 30
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="KEIROLABS_API_KEY",
                description="API key for Keirolabs (https://platform.keirolabs.cloud)",
                required=True,
            ),
        ]
    )

    def _run(self, **kwargs: Any) -> list[KeirolabsResult]:
        search_query: str | None = kwargs.get("search_query") or kwargs.get("query")
        if not search_query:
            raise ValueError("search_query is required")

        api_key = os.environ.get("KEIROLABS_API_KEY")
        if not api_key:
            raise ValueError("KEIROLABS_API_KEY environment variable is required")

        response = requests.post(
            f"{self.base_url}/api/v2/search/lite",
            json={"query": search_query, "maxResults": self.n_results},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        return [
            KeirolabsResult(
                title=result.get("title") or "",
                link=result["url"],
                snippet=result.get("snippet") or "",
            )
            for result in response.json().get("results", [])
            if result.get("url")
        ][: self.n_results]
