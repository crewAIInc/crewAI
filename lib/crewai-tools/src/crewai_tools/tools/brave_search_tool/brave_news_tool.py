from typing import Any

from pydantic import BaseModel

from crewai_tools.tools.brave_search_tool.base import BraveSearchToolBase
from crewai_tools.tools.brave_search_tool.schemas import (
    NewsSearchHeaders,
    NewsSearchParams,
)


class BraveNewsSearchTool(BraveSearchToolBase):
    """A tool that performs news searches using the Brave Search API."""

    name: str = "Brave News Search"
    args_schema: type[BaseModel] = NewsSearchParams
    header_schema: type[BaseModel] = NewsSearchHeaders

    description: str = (
        "A tool that performs news searches using the Brave Search API. "
        "Results are returned as structured JSON data."
    )

    search_url: str = "https://api.search.brave.com/res/v1/news/search"

    def _refine_request_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        return params

    def _refine_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # Make the response more concise, and easier to consume
        results = response.get("results", [])
        return [
            {
                "url": result.get("url"),
                "title": result.get("title"),
                "description": result.get("description"),
            }
            for result in results
        ]
