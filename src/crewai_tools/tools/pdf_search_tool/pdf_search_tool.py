from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool


class FixedPDFSearchToolSchema(BaseModel):
	"""Input for PDFSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the PDF's content")

class PDFSearchToolSchema(FixedPDFSearchToolSchema):
	"""Input for PDFSearchTool."""
	pdf: str = Field(..., description="Mandatory pdf path you want to search")

class PDFSearchTool(RagTool):
	name: str = "Search a PDF's content"
	description: str = "A tool that can be used to semantic search a query from a PDF's content."
	summarize: bool = False
	args_schema: Type[BaseModel] = PDFSearchToolSchema
	pdf: Optional[str] = None

	def __init__(self, pdf: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if pdf is not None:
			self.pdf = pdf
			self.description = f"A tool that can be used to semantic search a query the {pdf} PDF's content."
			self.args_schema = FixedPDFSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		pdf = kwargs.get('pdf', self.pdf)
		self.app = App()
		self.app.add(pdf, data_type=DataType.PDF_FILE)
		return super()._run(query=search_query)