from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool


class FixedWebsiteSearchToolSchema(BaseModel):
	"""Input for WebsiteSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search a specific website")

class WebsiteSearchToolSchema(FixedWebsiteSearchToolSchema):
	"""Input for WebsiteSearchTool."""
	website: str = Field(..., description="Mandatory valid website URL you want to search on")

class WebsiteSearchTool(RagTool):
	name: str = "Search in a specific website"
	description: str = "A tool that can be used to semantic search a query from a specific URL content."
	summarize: bool = False
	args_schema: Type[BaseModel] = WebsiteSearchToolSchema
	website: Optional[str] = None

	def __init__(self, website: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if website is not None:
			self.website = website
			self.description = f"A tool that can be used to semantic search a query from {website} website content."
			self.args_schema = FixedWebsiteSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		website = kwargs.get('website', self.website)
		self.app = App()
		self.app.add(website, data_type=DataType.WEB_PAGE)
		return super()._run(query=search_query)