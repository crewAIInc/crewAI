import logging
import os
from typing import Any, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class YouTubeSearchToolSchema(BaseModel):
    """Input for YouTubeSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search YouTube videos",
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return (default: 5, max: 50)",
        ge=1,
        le=50,
    )


class YouTubeSearchResult(BaseModel):
    """Schema for a single YouTube search result."""

    title: str
    video_id: str
    url: str
    description: str
    published_at: str


class YouTubeSearchTool(BaseTool):
    name: str = "Search YouTube Videos"
    description: str = (
        "A tool that searches YouTube videos using the YouTube Data API v3. "
        "Returns a list of videos with title, video_id, URL, description, and publish date."
    )
    args_schema: type[BaseModel] = YouTubeSearchToolSchema
    env_vars: list[dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "YOUTUBE_API_KEY",
                "description": "API key for YouTube Data API v3",
                "required": True,
            }
        ]
    )

    def _run(
        self,
        search_query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        """Execute the YouTube search operation.

        Args:
            search_query: The search query string
            max_results: Maximum number of results to return (1-50)

        Returns:
            List of dictionaries containing video information

        Raises:
            ValueError: If API key is not configured or invalid
            RuntimeError: If the API request fails
        """
        # Cap max_results to 50 to comply with YouTube API limits
        max_results = min(max_results, 50) 
      
        # Get API key from environment
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError(
                "YouTube API key not found. "
                "Please set the YOUTUBE_API_KEY environment variable. "
                "Get your API key from: https://console.cloud.google.com/apis/credentials"
            )

        try:
            # Import here to avoid hard dependency if not used
            from googleapiclient.discovery import build
            from googleapiclient.errors import HttpError
        except ImportError:
            raise ImportError(
                "google-api-python-client is not installed. "
                "Please install it with: pip install google-api-python-client google-auth"
            )

        try:
            # Build the YouTube API client
            youtube = build("youtube", "v3", developerKey=api_key)

            # Execute search request
            search_response = (
                youtube.search()
                .list(
                    q=search_query,
                    part="snippet",
                    type="video",
                    maxResults=max_results,
                    order="relevance",
                )
                .execute()
            )

            # Process results
            results: list[dict[str, str]] = []
            for item in search_response.get("items", []):
                snippet = item.get("snippet", {})
                video_id = item.get("id", {}).get("videoId", "")

                if video_id:
                    result = {
                        "title": snippet.get("title", ""),
                        "video_id": video_id,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "description": snippet.get("description", ""),
                        "published_at": snippet.get("publishedAt", ""),
                    }
                    results.append(result)

            logger.info(f"Found {len(results)} YouTube videos for query: {search_query}")
            return results

        except HttpError as e:
            error_msg = f"YouTube API error: {e.resp.status} - {e.content.decode('utf-8', errors='replace')}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Error searching YouTube: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
