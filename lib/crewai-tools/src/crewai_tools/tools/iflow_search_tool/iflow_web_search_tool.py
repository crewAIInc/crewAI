from __future__ import annotations

import json

from pydantic import BaseModel, Field

from crewai_tools.tools.iflow_search_tool.base import IFlowSearchToolBase


class IFlowWebSearchToolSchema(BaseModel):
    """Input schema for IFlowWebSearchTool."""

    query: str = Field(..., description="The search query string.")
    count: int | None = Field(
        default=None,
        description="Maximum number of results to return (server default when unset).",
    )


class IFlowWebSearchTool(IFlowSearchToolBase):
    """A tool that performs web searches using the iFlow Search API."""

    name: str = "iFlow Web Search"
    description: str = (
        "A tool that performs web searches using the iFlow Search API. "
        "Results are returned as structured JSON data."
    )
    args_schema: type[BaseModel] = IFlowWebSearchToolSchema

    def _run(self, query: str, count: int | None = None) -> str:
        response = self._get_client().web_search(query=query, count=count)
        return json.dumps(
            {
                "query": response.query,
                "results": [
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "position": result.position,
                        "date": result.date,
                    }
                    for result in response.results
                ],
            },
            ensure_ascii=False,
        )
