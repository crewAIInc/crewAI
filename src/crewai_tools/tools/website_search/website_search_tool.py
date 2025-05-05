from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedWebsiteSearchToolSchema(BaseModel):
    """Input for WebsiteSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search a specific website",
    )


class WebsiteSearchToolSchema(FixedWebsiteSearchToolSchema):
    """Input for WebsiteSearchTool."""

    website: str = Field(
        ..., description="Mandatory valid website URL you want to search on"
    )


class WebsiteSearchTool(RagTool):
    name: str = "Search in a specific website"
    description: str = (
        "A tool that can be used to semantic search a query from a specific URL content."
    )
    args_schema: Type[BaseModel] = WebsiteSearchToolSchema

    def __init__(self, website: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if website is not None:
            self.add(website)
            self.description = f"A tool that can be used to semantic search a query from {website} website content."
            self.args_schema = FixedWebsiteSearchToolSchema
            self._generate_description()

    def add(self, website: str) -> None:
        super().add(website, data_type=DataType.WEB_PAGE)

    def _run(
        self,
        search_query: str,
        website: Optional[str] = None,
    ) -> str:
        if website is not None:
            self.add(website)
        return super()._run(query=search_query)
