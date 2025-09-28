from typing import Any, Type

from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool
from crewai_tools.rag.data_types import DataType


class MySQLSearchToolSchema(BaseModel):
    """Input for MySQLSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory semantic search query you want to use to search the database's content",
    )


class MySQLSearchTool(RagTool):
    name: str = "Search a database's table content"
    description: str = "A tool that can be used to semantic search a query from a database table's content."
    args_schema: Type[BaseModel] = MySQLSearchToolSchema
    db_uri: str = Field(..., description="Mandatory database URI")

    def __init__(self, table_name: str, **kwargs):
        super().__init__(**kwargs)
        self.add(table_name, data_type=DataType.MYSQL, metadata={"db_uri": self.db_uri})
        self.description = f"A tool that can be used to semantic search a query the {table_name} database table's content."
        self._generate_description()

    def add(
        self,
        table_name: str,
        **kwargs: Any,
    ) -> None:
        super().add(f"SELECT * FROM {table_name};", **kwargs)

    def _run(
        self,
        search_query: str,
        similarity_threshold: float | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Any:
        return super()._run(query=search_query, similarity_threshold=similarity_threshold, limit=limit)
