from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool


class FixedYoutubeVideoSearchToolSchema(BaseModel):
	"""Input for YoutubeVideoSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the Youtube Video content")

class YoutubeVideoSearchToolSchema(FixedYoutubeVideoSearchToolSchema):
	"""Input for YoutubeVideoSearchTool."""
	youtube_video_url: str = Field(..., description="Mandatory youtube_video_url path you want to search")

class YoutubeVideoSearchTool(RagTool):
	name: str = "Search a Youtube Video content"
	description: str = "A tool that can be used to semantic search a query from a Youtube Video content."
	summarize: bool = False
	args_schema: Type[BaseModel] = YoutubeVideoSearchToolSchema
	youtube_video_url: Optional[str] = None

	def __init__(self, youtube_video_url: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if youtube_video_url is not None:
			self.youtube_video_url = youtube_video_url
			self.description = f"A tool that can be used to semantic search a query the {youtube_video_url} Youtube Video content."
			self.args_schema = FixedYoutubeVideoSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		youtube_video_url = kwargs.get('youtube_video_url', self.youtube_video_url)
		self.app = App()
		self.app.add(youtube_video_url, data_type=DataType.YOUTUBE_VIDEO)
		return super()._run(query=search_query)