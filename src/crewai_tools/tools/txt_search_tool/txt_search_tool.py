from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool

class FixedTXTSearchToolSchema(BaseModel):
	"""Input for TXTSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the txt's content")

class TXTSearchToolSchema(FixedTXTSearchToolSchema):
	"""Input for TXTSearchTool."""
	txt: str = Field(..., description="Mandatory txt path you want to search")

class TXTSearchTool(RagTool):
	name: str = "Search a txt's content"
	description: str = "A tool that can be used to semantic search a query from a txt's content."
	summarize: bool = False
	args_schema: Type[BaseModel] = TXTSearchToolSchema
	txt: Optional[str] = None

	def __init__(self, txt: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if txt is not None:
			self.txt = txt
			self.description = f"A tool that can be used to semantic search a query the {txt} txt's content."
			self.args_schema = FixedTXTSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		txt = kwargs.get('txt', self.txt)
		self.app = App()
		self.app.add(txt, data_type=DataType.TEXT_FILE)
		return super()._run(query=search_query)