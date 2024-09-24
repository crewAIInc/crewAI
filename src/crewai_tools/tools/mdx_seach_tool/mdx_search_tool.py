from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedMDXSearchToolSchema(BaseModel):
    """Input for MDXSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the MDX's content",
    )


class MDXSearchToolSchema(FixedMDXSearchToolSchema):
    """Input for MDXSearchTool."""

    mdx: str = Field(..., description="Mandatory mdx path you want to search")


class MDXSearchTool(RagTool):
    name: str = "Search a MDX's content"
    description: str = (
        "A tool that can be used to semantic search a query from a MDX's content."
    )
    args_schema: Type[BaseModel] = MDXSearchToolSchema

    def __init__(self, mdx: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if mdx is not None:
            kwargs["data_type"] = DataType.MDX
            self.add(mdx)
            self.description = f"A tool that can be used to semantic search a query the {mdx} MDX's content."
            self.args_schema = FixedMDXSearchToolSchema
            self._generate_description()

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().add(*args, **kwargs)

    def _before_run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        if "mdx" in kwargs:
            self.add(kwargs["mdx"])

    def _run(
        self,
        search_query: str,
        **kwargs: Any,
    ) -> Any:
        return super()._run(query=search_query, **kwargs)
