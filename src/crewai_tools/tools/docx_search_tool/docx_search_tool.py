from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool


class FixedDOCXSearchToolSchema(BaseModel):
	"""Input for DOCXSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the DOCX's content")

class DOCXSearchToolSchema(FixedDOCXSearchToolSchema):
	"""Input for DOCXSearchTool."""
	docx: str = Field(..., description="Mandatory docx path you want to search")

class DOCXSearchTool(RagTool):
	name: str = "Search a DOCX's content"
	description: str = "A tool that can be used to semantic search a query from a DOCX's content."
	summarize: bool = False
	args_schema: Type[BaseModel] = DOCXSearchToolSchema
	docx: Optional[str] = None

	def __init__(self, docx: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if docx is not None:
			self.docx = docx
			self.description = f"A tool that can be used to semantic search a query the {docx} DOCX's content."
			self.args_schema = FixedDOCXSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		docx = kwargs.get('docx', self.docx)
		self.app = App()
		self.app.add(docx, data_type=DataType.DOCX)
		return super()._run(query=search_query)