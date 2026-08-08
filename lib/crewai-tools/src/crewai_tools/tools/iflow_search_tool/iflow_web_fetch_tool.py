from __future__ import annotations

import json

from pydantic import BaseModel, Field

from crewai_tools.security.safe_path import validate_url
from crewai_tools.tools.iflow_search_tool.base import IFlowSearchToolBase


class IFlowWebFetchToolSchema(BaseModel):
    """Input schema for IFlowWebFetchTool."""

    url: str = Field(..., description="The URL of the web page to fetch.")


class IFlowWebFetchTool(IFlowSearchToolBase):
    """A tool that fetches and extracts web page content using the iFlow Search API."""

    name: str = "iFlow Web Fetch"
    description: str = (
        "A tool that fetches a web page and extracts its content using the "
        "iFlow Search API. Returns the page title and extracted text as JSON."
    )
    args_schema: type[BaseModel] = IFlowWebFetchToolSchema

    def _run(self, url: str) -> str:
        """Fetch a web page via iFlow and return normalized JSON content."""
        validated_url = validate_url(url)
        response = self._get_client().web_fetch(url=validated_url)
        return json.dumps(
            {
                "url": response.url,
                "title": response.title,
                "content": response.content,
                "from_cache": response.from_cache,
            },
            ensure_ascii=False,
        )
