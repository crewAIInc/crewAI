from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


class FixedDOCXSearchToolSchema(BaseModel):
    """Input for DOCXSearchTool."""

    docx: str | None = Field(
        ..., description="File path or URL of a DOCX file to be searched"
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
    args_schema: type[BaseModel] = DOCXSearchToolSchema

    def __init__(self, docx: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if docx is not None:
            self.add(docx)
            self.description = f"A tool that can be used to semantic search a query the {docx} DOCX's content."
            self.args_schema = FixedDOCXSearchToolSchema
            self._generate_description()

    def add(self, docx: str) -> None:
        super().add(docx, data_type=DataType.DOCX)

    def _run(  # type: ignore[override]
        self,
        search_query: str,
        docx: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> Any:
        if docx is not None:
            self.add(docx)
        return super()._run(
            query=search_query, similarity_threshold=similarity_threshold, limit=limit
        )
