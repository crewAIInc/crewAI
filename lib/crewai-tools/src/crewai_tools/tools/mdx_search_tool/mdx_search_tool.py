from pydantic import BaseModel, Field

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


class FixedMDXSearchToolSchema(BaseModel):
    """Input for MDXSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the MDX's content",
    )


class MDXSearchToolSchema(FixedMDXSearchToolSchema):
    """Input for MDXSearchTool."""

    mdx: str = Field(..., description="File path or URL of a MDX file to be searched")


class MDXSearchTool(RagTool):
    name: str = "Search a MDX's content"
    description: str = (
        "A tool that can be used to semantic search a query from a MDX's content."
    )
    args_schema: type[BaseModel] = MDXSearchToolSchema

    def __init__(self, mdx: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if mdx is not None:
            self.add(mdx)
            self.description = f"A tool that can be used to semantic search a query the {mdx} MDX's content."
            self.args_schema = FixedMDXSearchToolSchema
            self._generate_description()

    def add(self, mdx: str) -> None:
        super().add(mdx, data_type=DataType.MDX)

    def _run(  # type: ignore[override]
        self,
        search_query: str,
        mdx: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        if mdx is not None:
            self.add(mdx)
        return super()._run(
            query=search_query, similarity_threshold=similarity_threshold, limit=limit
        )
