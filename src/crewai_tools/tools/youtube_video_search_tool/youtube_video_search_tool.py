from typing import Any, Optional, Type

from embedchain.models.data_type import DataType
from pydantic.v1 import BaseModel, Field

from ..rag.rag_tool import RagTool


class FixedYoutubeVideoSearchToolSchema(BaseModel):
    """Input for YoutubeVideoSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the Youtube Video content",
    )


class YoutubeVideoSearchToolSchema(FixedYoutubeVideoSearchToolSchema):
    """Input for YoutubeVideoSearchTool."""

    youtube_video_url: str = Field(
        ..., description="Mandatory youtube_video_url path you want to search"
    )


class YoutubeVideoSearchTool(RagTool):
    name: str = "Search a Youtube Video content"
    description: str = "A tool that can be used to semantic search a query from a Youtube Video content."
    args_schema: Type[BaseModel] = YoutubeVideoSearchToolSchema

    def __init__(self, youtube_video_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if youtube_video_url is not None:
            self.add(youtube_video_url)
            self.description = f"A tool that can be used to semantic search a query the {youtube_video_url} Youtube Video content."
            self.args_schema = FixedYoutubeVideoSearchToolSchema
            self._generate_description()

    def add(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        kwargs["data_type"] = DataType.YOUTUBE_VIDEO
        super().add(*args, **kwargs)

    def _before_run(
        self,
        query: str,
        **kwargs: Any,
    ) -> Any:
        if "youtube_video_url" in kwargs:
            self.add(kwargs["youtube_video_url"])

    def _run(
        self,
        search_query: str,
        **kwargs: Any,
    ) -> Any:
        return super()._run(query=search_query)
