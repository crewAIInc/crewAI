from typing import Any, Type

try:
    from embedchain.loaders.postgres import PostgresLoader
    EMBEDCHAIN_AVAILABLE = True
except ImportError:
    EMBEDCHAIN_AVAILABLE = False

from pydantic import BaseModel, Field

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
        if not EMBEDCHAIN_AVAILABLE:
            raise ImportError("embedchain is not installed. Please install it with `pip install crewai-tools[embedchain]`")
        super().__init__(**kwargs)
        kwargs["data_type"] = "postgres"
        kwargs["loader"] = PostgresLoader(config=dict(url=self.db_uri))
        self.add(table_name)
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
        **kwargs: Any,
    ) -> Any:
        return super()._run(query=search_query, **kwargs)
