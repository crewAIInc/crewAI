from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedCodeDocsSearchToolSchema(BaseModel):
    """Input for CodeDocsSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the Code Docs content",
    )


class CodeDocsSearchToolSchema(FixedCodeDocsSearchToolSchema):
    """Input for CodeDocsSearchTool."""

    docs_url: str = Field(..., description="Mandatory docs_url path you want to search")


class CodeDocsSearchTool(RagTool):
    name: str = "Search a Code Docs content"
    description: str = (
        "A tool that can be used to semantic search a query from a Code Docs content."
    )
    args_schema: Type[BaseModel] = CodeDocsSearchToolSchema

    def __init__(self, docs_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if docs_url is not None:
            self.add(docs_url)
            self.description = f"A tool that can be used to semantic search a query the {docs_url} Code Docs content."
            self.args_schema = FixedCodeDocsSearchToolSchema
            self._generate_description()

    def add(self, docs_url: str) -> None:
        super().add(docs_url, data_type=DataType.DOCS_SITE)

    def _run(
        self,
        search_query: str,
        docs_url: Optional[str] = None,
    ) -> str:
        if docs_url is not None:
            self.add(docs_url)
        return super()._run(query=search_query)
