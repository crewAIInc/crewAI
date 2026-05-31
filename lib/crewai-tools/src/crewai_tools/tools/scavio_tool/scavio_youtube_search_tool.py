from typing import Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.scavio_tool.base import ScavioBaseTool


class ScavioYouTubeSearchToolSchema(BaseModel):
    """Input schema for ScavioYouTubeSearchTool."""

    query: str = Field(..., description="YouTube search query.")


class ScavioYouTubeSearchTool(ScavioBaseTool):
    """Tool that searches YouTube videos using the Scavio API.

    Attributes:
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The Scavio API key.
        max_results: The maximum number of results to return.
        upload_date: Filter by upload date.
        sort_by: Sort order for results.
        type: Type of YouTube content to search for.
        duration: Filter by video duration.
    """

    name: str = "Scavio YouTube Search"
    description: str = (
        "A tool that searches YouTube videos using the Scavio API. "
        "It returns video titles, URLs, view counts, channel information, "
        "and thumbnails as a JSON string. Use for finding videos on any topic."
    )
    args_schema: type[BaseModel] = ScavioYouTubeSearchToolSchema

    upload_date: Literal["hour", "today", "week", "month", "year"] | None = Field(
        default=None,
        description="Filter results by upload date.",
    )
    sort_by: Literal["relevance", "date", "views", "rating"] | None = Field(
        default=None,
        description="Sort order for results.",
    )
    type: Literal["video", "channel", "playlist"] | None = Field(
        default=None,
        description="Type of YouTube content to search for.",
    )
    duration: Literal["short", "medium", "long"] | None = Field(
        default=None,
        description="Filter by video duration (short < 4min, medium 4-20min, long > 20min).",
    )

    def _run(self, query: str) -> str:
        """Synchronously searches YouTube videos.

        Args:
            query: YouTube search query.

        Returns:
            A JSON string containing YouTube search results.
        """
        if not self.client:
            raise ValueError(
                "Scavio client is not initialized. Ensure 'scavio' is "
                "installed and API key is set."
            )

        raw = self.client.youtube.search(
            query=query,
            upload_date=self.upload_date,
            sort_by=self.sort_by,
            type=self.type,
            duration=self.duration,
        )

        data = raw.get("data")
        if isinstance(data, list) and len(data) > self.max_results:
            raw["data"] = data[: self.max_results]

        return self._format_response(raw)

    async def _arun(self, query: str) -> str:
        """Asynchronously searches YouTube videos.

        Args:
            query: YouTube search query.

        Returns:
            A JSON string containing YouTube search results.
        """
        if not self.async_client:
            raise ValueError(
                "Scavio async client is not initialized. Ensure 'scavio' "
                "is installed and API key is set."
            )

        raw = await self.async_client.youtube.search(
            query=query,
            upload_date=self.upload_date,
            sort_by=self.sort_by,
            type=self.type,
            duration=self.duration,
        )

        data = raw.get("data")
        if isinstance(data, list) and len(data) > self.max_results:
            raw["data"] = data[: self.max_results]

        return self._format_response(raw)
