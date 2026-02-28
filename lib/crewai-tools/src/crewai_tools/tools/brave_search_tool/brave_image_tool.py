from typing import Any

from pydantic import BaseModel

from crewai_tools.tools.brave_search_tool.base import BraveSearchToolBase
from crewai_tools.tools.brave_search_tool.schemas import (
    ImageSearchHeaders,
    ImageSearchParams,
)


class BraveImageSearchTool(BraveSearchToolBase):
    """A tool that performs image searches using the Brave Search API."""

    name: str = "Brave Image Search"
    args_schema: type[BaseModel] = ImageSearchParams
    header_schema: type[BaseModel] = ImageSearchHeaders

    description: str = (
        "A tool that performs image searches using the Brave Search API. "
        "Results are returned as structured JSON data."
    )

    search_url: str = "https://api.search.brave.com/res/v1/images/search"

    def _refine_request_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        return params

    def _refine_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # Make the response more concise, and easier to consume
        results = response.get("results", [])
        return [
            {
                "title": result.get("title"),
                "url": result.get("properties", {}).get("url"),
                "dimensions": f"{w}x{h}"
                if (w := result.get("properties", {}).get("width"))
                and (h := result.get("properties", {}).get("height"))
                else None,
            }
            for result in results
        ]
