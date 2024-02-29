from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool


class FixedCSVSearchToolSchema(BaseModel):
	"""Input for CSVSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the CSV's content")

class CSVSearchToolSchema(FixedCSVSearchToolSchema):
	"""Input for CSVSearchTool."""
	pdf: str = Field(..., description="Mandatory csv path you want to search")

class CSVSearchTool(RagTool):
	name: str = "Search a CSV's content"
	description: str = "A tool that can be used to semantic search a query from a CSV's content."
	summarize: bool = False
	args_schema: Type[BaseModel] = CSVSearchToolSchema
	csv: Optional[str] = None

	def __init__(self, csv: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if csv is not None:
			self.csv = csv
			self.description = f"A tool that can be used to semantic search a query the {csv} CSV's content."
			self.args_schema = FixedCSVSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		csv = kwargs.get('csv', self.csv)
		self.app = App()
		self.app.add(csv, data_type=DataType.CSV)
		return super()._run(query=search_query)