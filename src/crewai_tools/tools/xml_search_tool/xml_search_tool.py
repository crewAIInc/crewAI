from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool


class FixedXMLSearchToolSchema(BaseModel):
	"""Input for XMLSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the XML's content")

class XMLSearchToolSchema(FixedXMLSearchToolSchema):
	"""Input for XMLSearchTool."""
	xml: str = Field(..., description="Mandatory xml path you want to search")

class XMLSearchTool(RagTool):
	name: str = "Search a XML's content"
	description: str = "A tool that can be used to semantic search a query from a XML's content."
	summarize: bool = False
	args_schema: Type[BaseModel] = XMLSearchToolSchema
	xml: Optional[str] = None

	def __init__(self, xml: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if xml is not None:
			self.xml = xml
			self.description = f"A tool that can be used to semantic search a query the {xml} XML's content."
			self.args_schema = FixedXMLSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		xml = kwargs.get('xml', self.xml)
		self.app = App()
		self.app.add(xml, data_type=DataType.XML)
		return super()._run(query=search_query)