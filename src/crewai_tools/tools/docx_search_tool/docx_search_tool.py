from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic.v1 import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedDOCXSearchToolSchema(BaseModel):
    """Input for DOCXSearchTool."""
    docx: Optional[str] = Field(..., description="Mandatory docx path you want to search")
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

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs["data_type"] = DataType.DOCX
        super().add(*args, **kwargs)

    def _before_run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        if "docx" in kwargs:
            self.add(kwargs["docx"])

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        search_query = kwargs.get('search_query')
        if search_query is None:
            search_query = kwargs.get('query')

        docx = kwargs.get("docx")
        if docx is not None:
            self.add(docx)
        return super()._run(query=search_query)
