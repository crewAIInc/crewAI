from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool


class FixedCodeDocsSearchToolSchema(BaseModel):
	"""Input for CodeDocsSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the Code Docs content")

class CodeDocsSearchToolSchema(FixedCodeDocsSearchToolSchema):
	"""Input for CodeDocsSearchTool."""
	docs_url: str = Field(..., description="Mandatory docs_url path you want to search")

class CodeDocsSearchTool(RagTool):
	name: str = "Search a Code Docs content"
	description: str = "A tool that can be used to semantic search a query from a Code Docs content."
	summarize: bool = False
	args_schema: Type[BaseModel] = CodeDocsSearchToolSchema
	docs_url: Optional[str] = None

	def __init__(self, docs_url: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if docs_url is not None:
			self.docs_url = docs_url
			self.description = f"A tool that can be used to semantic search a query the {docs_url} Code Docs content."
			self.args_schema = FixedCodeDocsSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		docs_url = kwargs.get('docs_url', self.docs_url)
		self.app = App()
		self.app.add(docs_url, data_type=DataType.DOCS_SITE)
		return super()._run(query=search_query)