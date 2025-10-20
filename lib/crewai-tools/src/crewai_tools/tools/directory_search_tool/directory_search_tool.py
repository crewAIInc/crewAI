from pydantic import BaseModel, Field

from crewai_tools.rag.data_types import DataType
from crewai_tools.tools.rag.rag_tool import RagTool


class FixedDirectorySearchToolSchema(BaseModel):
    """Input for DirectorySearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the directory's content",
    )


class DirectorySearchToolSchema(FixedDirectorySearchToolSchema):
    """Input for DirectorySearchTool."""

    directory: str = Field(..., description="Mandatory directory you want to search")


class DirectorySearchTool(RagTool):
    name: str = "Search a directory's content"
    description: str = (
        "A tool that can be used to semantic search a query from a directory's content."
    )
    args_schema: type[BaseModel] = DirectorySearchToolSchema

    def __init__(self, directory: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if directory is not None:
            self.add(directory)
            self.description = f"A tool that can be used to semantic search a query the {directory} directory's content."
            self.args_schema = FixedDirectorySearchToolSchema
            self._generate_description()

    def add(self, directory: str) -> None:
        super().add(directory, data_type=DataType.DIRECTORY)

    def _run(  # type: ignore[override]
        self,
        search_query: str,
        directory: str | None = None,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        if directory is not None:
            self.add(directory)
        return super()._run(
            query=search_query, similarity_threshold=similarity_threshold, limit=limit
        )
