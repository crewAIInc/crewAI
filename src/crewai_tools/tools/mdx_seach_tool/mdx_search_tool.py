from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool


class FixedMDXSearchToolSchema(BaseModel):
	"""Input for MDXSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the MDX's content")

class MDXSearchToolSchema(FixedMDXSearchToolSchema):
	"""Input for MDXSearchTool."""
	mdx: str = Field(..., description="Mandatory mdx path you want to search")

class MDXSearchTool(RagTool):
	name: str = "Search a MDX's content"
	description: str = "A tool that can be used to semantic search a query from a MDX's content."
	summarize: bool = False
	args_schema: Type[BaseModel] = MDXSearchToolSchema
	mdx: Optional[str] = None

	def __init__(self, mdx: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if mdx is not None:
			self.mdx = mdx
			self.description = f"A tool that can be used to semantic search a query the {mdx} MDX's content."
			self.args_schema = FixedMDXSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		mdx = kwargs.get('mdx', self.mdx)
		self.app = App()
		self.app.add(mdx, data_type=DataType.MDX)
		return super()._run(query=search_query)