from typing import Any

from pydantic import BaseModel

from crewai_tools.tools.brave_search_tool.base import BraveSearchToolBase
from crewai_tools.tools.brave_search_tool.schemas import (
    WebSearchHeaders,
    WebSearchParams,
)


class BraveWebSearchTool(BraveSearchToolBase):
    """A tool that performs web searches using the Brave Search API."""

    name: str = "Brave Web Search"
    args_schema: type[BaseModel] = WebSearchParams
    header_schema: type[BaseModel] = WebSearchHeaders

    description: str = (
        "A tool that performs web searches using the Brave Search API. "
        "Results are returned as structured JSON data."
    )

    search_url: str = "https://api.search.brave.com/res/v1/web/search"

    def _refine_request_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        return params

    def _refine_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # Make the response more concise, and easier to consume
        results = response.get("web", {}).get("results", [])
        return [
            {
                "url": result.get("url"),
                "title": result.get("title"),
                "description": result.get("extra_snippets")
                or result.get("description"),
            }
            for result in results
        ]
