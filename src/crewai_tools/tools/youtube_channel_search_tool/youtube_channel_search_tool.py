from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedYoutubeChannelSearchToolSchema(BaseModel):
    """Input for YoutubeChannelSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the Youtube Channels content",
    )


class YoutubeChannelSearchToolSchema(FixedYoutubeChannelSearchToolSchema):
    """Input for YoutubeChannelSearchTool."""

    youtube_channel_handle: str = Field(
        ..., description="Mandatory youtube_channel_handle path you want to search"
    )


class YoutubeChannelSearchTool(RagTool):
    name: str = "Search a Youtube Channels content"
    description: str = (
        "A tool that can be used to semantic search a query from a Youtube Channels content."
    )
    args_schema: Type[BaseModel] = YoutubeChannelSearchToolSchema

    def __init__(self, youtube_channel_handle: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if youtube_channel_handle is not None:
            self.add(youtube_channel_handle)
            self.description = f"A tool that can be used to semantic search a query the {youtube_channel_handle} Youtube Channels content."
            self.args_schema = FixedYoutubeChannelSearchToolSchema
            self._generate_description()

    def add(
        self,
        youtube_channel_handle: str,
        **kwargs: Any,
    ) -> None:
        if not youtube_channel_handle.startswith("@"):
            youtube_channel_handle = f"@{youtube_channel_handle}"

        kwargs["data_type"] = DataType.YOUTUBE_CHANNEL
        super().add(youtube_channel_handle, **kwargs)

    def _before_run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        if "youtube_channel_handle" in kwargs:
            self.add(kwargs["youtube_channel_handle"])

    def _run(
        self,
        search_query: str,
        **kwargs: Any,
    ) -> Any:
        return super()._run(query=search_query, **kwargs)
