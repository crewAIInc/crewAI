from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedXMLSearchToolSchema(BaseModel):
    """Input for XMLSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the XML's content",
    )


class XMLSearchToolSchema(FixedXMLSearchToolSchema):
    """Input for XMLSearchTool."""

    xml: str = Field(..., description="Mandatory xml path you want to search")


class XMLSearchTool(RagTool):
    name: str = "Search a XML's content"
    description: str = (
        "A tool that can be used to semantic search a query from a XML's content."
    )
    args_schema: Type[BaseModel] = XMLSearchToolSchema

    def __init__(self, xml: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if xml is not None:
            self.add(xml)
            self.description = f"A tool that can be used to semantic search a query the {xml} XML's content."
            self.args_schema = FixedXMLSearchToolSchema
            self._generate_description()

    def _run(
        self,
        search_query: str,
        xml: Optional[str] = None,
    ) -> str:
        if xml is not None:
            self.add(xml)
        return super()._run(query=search_query)
