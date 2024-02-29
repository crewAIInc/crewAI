from typing import Optional, Type, Any
from pydantic.v1 import BaseModel, Field

from embedchain import App
from embedchain.models.data_type import DataType

from ..rag.rag_tool import RagTool


class FixedYoutubeChannelSearchToolSchema(BaseModel):
	"""Input for YoutubeChannelSearchTool."""
	search_query: str = Field(..., description="Mandatory search query you want to use to search the Youtube Channels content")

class YoutubeChannelSearchToolSchema(FixedYoutubeChannelSearchToolSchema):
	"""Input for YoutubeChannelSearchTool."""
	youtube_channel_handle: str = Field(..., description="Mandatory youtube_channel_handle path you want to search")

class YoutubeChannelSearchTool(RagTool):
	name: str = "Search a Youtube Channels content"
	description: str = "A tool that can be used to semantic search a query from a Youtube Channels content."
	summarize: bool = False
	args_schema: Type[BaseModel] = YoutubeChannelSearchToolSchema
	youtube_channel_handle: Optional[str] = None

	def __init__(self, youtube_channel_handle: Optional[str] = None, **kwargs):
		super().__init__(**kwargs)
		if youtube_channel_handle is not None:
			self.youtube_channel_handle = youtube_channel_handle
			self.description = f"A tool that can be used to semantic search a query the {youtube_channel_handle} Youtube Channels content."
			self.args_schema = FixedYoutubeChannelSearchToolSchema
			self._generate_description()

	def _run(
		self,
		search_query: str,
		**kwargs: Any,
	) -> Any:
		youtube_channel_handle = kwargs.get('youtube_channel_handle', self.youtube_channel_handle)
		if not youtube_channel_handle.startswith("@"):
			youtube_channel_handle = f"@{youtube_channel_handle}"
		self.app = App()
		self.app.add(youtube_channel_handle, data_type=DataType.YOUTUBE_CHANNEL)
		return super()._run(query=search_query)