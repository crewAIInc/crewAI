from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedJSONSearchToolSchema(BaseModel):
    """Input for JSONSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the JSON's content",
    )


class JSONSearchToolSchema(FixedJSONSearchToolSchema):
    """Input for JSONSearchTool."""

    json_path: str = Field(..., description="Mandatory json path you want to search")


class JSONSearchTool(RagTool):
    name: str = "Search a JSON's content"
    description: str = (
        "A tool that can be used to semantic search a query from a JSON's content."
    )
    args_schema: Type[BaseModel] = JSONSearchToolSchema

    def __init__(self, json_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if json_path is not None:
            self.add(json_path)
            self.description = f"A tool that can be used to semantic search a query the {json_path} JSON's content."
            self.args_schema = FixedJSONSearchToolSchema
            self._generate_description()

    def _run(
        self,
        search_query: str,
        json_path: Optional[str] = None,
    ) -> str:
        if json_path is not None:
            self.add(json_path)
        return super()._run(query=search_query)
