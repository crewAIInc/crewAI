from typing import Any, Type

from embedchain.loaders.postgres import PostgresLoader
from pydantic.v1 import BaseModel, Field

from ..rag.rag_tool import RagTool


class PGSearchToolSchema(BaseModel):
    """Input for PGSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory semantic search query you want to use to search the database's content",
    )


class PGSearchTool(RagTool):
    name: str = "Search a database's table content"
    description: str = "A tool that can be used to semantic search a query from a database table's content."
    args_schema: Type[BaseModel] = PGSearchToolSchema
    db_uri: str = Field(..., description="Mandatory database URI")

    def __init__(self, table_name: str, **kwargs):
        super().__init__(**kwargs)
        self.add(table_name)
        self.description = f"A tool that can be used to semantic search a query the {table_name} database table's content."
        self._generate_description()

    def add(
        self,
        table_name: str,
        **kwargs: Any,
    ) -> None:
        kwargs["data_type"] = "postgres"
        kwargs["loader"] = PostgresLoader(config=dict(url=self.db_uri))
        super().add(f"SELECT * FROM {table_name};", **kwargs)

    def _run(
        self,
        search_query: str,
        **kwargs: Any,
    ) -> Any:
        return super()._run(query=search_query)
