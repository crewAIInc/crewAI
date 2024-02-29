from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.loaders.postgres import PostgresLoader

from ..rag.rag_tool import RagTool

class PGSearchToolSchema(BaseModel):
	"""Input for PGSearchTool."""
	search_query: str = Field(..., description="Mandatory semantic search query you want to use to search the database's content")

class PGSearchTool(RagTool):
	name: str = "Search a database's table content"
	description: str = "A tool that can be used to semantic search a query from a database table's content."
	summarize: bool = False
	args_schema: Type[BaseModel] = PGSearchToolSchema
	db_uri: str = Field(..., description="Mandatory database URI")
	table_name: str = Field(..., description="Mandatory table name")
	search_query: str = Field(..., description="Mandatory semantic search query you want to use to search the database's content")

	def __init__(self, table_name: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if table_name is not None:
			self.table_name = table_name
			self.description = f"A tool that can be used to semantic search a query the {table_name} database table's content."
			self._generate_description()
		else:
			raise('To use PGSearchTool, you must provide a `table_name` argument')

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:

		config = { "url":  self.db_uri }
		postgres_loader = PostgresLoader(config=config)
		app = App()
		app.add(
				f"SELECT * FROM {self.table_name};",
				data_type='postgres',
				loader=postgres_loader
		)
		return super()._run(query=search_query)