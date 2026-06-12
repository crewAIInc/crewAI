from __future__ import annotations

import json

from pydantic import BaseModel, Field

from crewai_tools.tools.iflow_search_tool.base import IFlowSearchToolBase


class IFlowImageSearchToolSchema(BaseModel):
    """Input schema for IFlowImageSearchTool."""

    query: str = Field(..., description="The image search query string.")
    count: int | None = Field(
        default=None,
        description="Maximum number of images to return (server default when unset).",
    )


class IFlowImageSearchTool(IFlowSearchToolBase):
    """A tool that performs image searches using the iFlow Search API."""

    name: str = "iFlow Image Search"
    description: str = (
        "A tool that performs image searches using the iFlow Search API. "
        "Results are returned as structured JSON data with image and source URLs."
    )
    args_schema: type[BaseModel] = IFlowImageSearchToolSchema

    def _run(self, query: str, count: int | None = None) -> str:
        response = self._get_client().image_search(query=query, count=count)
        return json.dumps(
            {
                "query": response.query,
                "images": [
                    {
                        "image_url": image.image_url,
                        "source_url": image.source_url,
                        "title": image.title,
                        "width": image.width,
                        "height": image.height,
                        "position": image.position,
                    }
                    for image in response.images
                ],
            },
            ensure_ascii=False,
        )
