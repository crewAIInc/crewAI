from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedDOCXSearchToolSchema(BaseModel):
    """Input for DOCXSearchTool."""

    docx: Optional[str] = Field(
        ..., description="Mandatory docx path you want to search"
    )
    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the DOCX's content",
    )


class DOCXSearchToolSchema(FixedDOCXSearchToolSchema):
    """Input for DOCXSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the DOCX's content",
    )


class DOCXSearchTool(RagTool):
    name: str = "Search a DOCX's content"
    description: str = (
        "A tool that can be used to semantic search a query from a DOCX's content."
    )
    args_schema: Type[BaseModel] = DOCXSearchToolSchema

    def __init__(self, docx: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if docx is not None:
            self.add(docx)
            self.description = f"A tool that can be used to semantic search a query the {docx} DOCX's content."
            self.args_schema = FixedDOCXSearchToolSchema
            self._generate_description()

    def add(self, docx: str) -> None:
        super().add(docx, data_type=DataType.DOCX)

    def _run(
        self,
        search_query: str,
        docx: Optional[str] = None,
    ) -> Any:
        if docx is not None:
            self.add(docx)
        return super()._run(query=search_query)
