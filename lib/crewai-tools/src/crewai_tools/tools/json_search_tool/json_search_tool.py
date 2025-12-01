from pydantic import BaseModel, Field

from crewai_tools.tools.rag.rag_tool import RagTool


class FixedJSONSearchToolSchema(BaseModel):
    """Input for JSONSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the JSON's content",
    )


class JSONSearchToolSchema(FixedJSONSearchToolSchema):
    """Input for JSONSearchTool."""

    json_path: str = Field(
        ..., description="File path or URL of a JSON file to be searched"
    )


class JSONSearchTool(RagTool):
    name: str = "Search a JSON's content"
    description: str = (
        "A tool that can be used to semantic search a query from a JSON's content."
    )
    args_schema: type[BaseModel] = JSONSearchToolSchema

    def __init__(self, json_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if json_path is not None:
            self.add(json_path)
            self.description = f"A tool that can be used to semantic search a query the {json_path} JSON's content."
            self.args_schema = FixedJSONSearchToolSchema
            self._generate_description()

    def _run(  # type: ignore[override]
        self,
        search_query: str,
        json_path: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        if json_path is not None:
            self.add(json_path)
        return super()._run(
            query=search_query, similarity_threshold=similarity_threshold, limit=limit
        )
