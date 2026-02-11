from typing import Any

from pydantic import BaseModel

from crewai_tools.tools.brave_search_tool.base import BraveSearchToolBase
from crewai_tools.tools.brave_search_tool.schemas import (
    VideoSearchHeaders,
    VideoSearchParams,
)


class BraveVideoSearchTool(BraveSearchToolBase):
    """A tool that performs video searches using the Brave Search API."""

    name: str = "Brave Video Search"
    args_schema: type[BaseModel] = VideoSearchParams
    header_schema: type[BaseModel] = VideoSearchHeaders

    description: str = (
        "A tool that performs video searches using the Brave Search API. "
        "Results are returned as structured JSON data."
    )

    search_url: str = "https://api.search.brave.com/res/v1/videos/search"

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
