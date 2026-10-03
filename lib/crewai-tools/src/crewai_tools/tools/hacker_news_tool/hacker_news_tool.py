import requests
from typing import Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class HackerNewsTopStoriesToolInput(BaseModel):
    """Input schema for HackerNewsTopStoriesTool."""

    limit: int = Field(
        default=5,
        description="Number of top stories to retrieve (1 to 20).",
        ge=1,
        le=20,
    )


class HackerNewsTopStoriesTool(BaseTool):
    name: str = "Hacker News Top Stories"
    description: str = (
        "Fetches current top stories from Hacker News, returning story titles, "
        "scores, and URLs."
    )
    args_schema: Type[BaseModel] = HackerNewsTopStoriesToolInput

    def _run(self, limit: int = 5) -> str:
        """Fetch and format the current top Hacker News stories."""
        limit = max(1, min(limit, 20))
        top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"

        try:
            response = requests.get(top_stories_url, timeout=10)
            response.raise_for_status()
            story_ids = response.json()

            stories = []

            for story_id in story_ids:
                item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                item_res = requests.get(item_url, timeout=10)

                if item_res.status_code != 200:
                    continue

                item = item_res.json()

                if (
                    not item
                    or not isinstance(item, dict)
                    or item.get("type") != "story"
                    or item.get("deleted")
                    or item.get("dead")
                ):
                    continue

                title = item.get("title", "Untitled")
                url = item.get(
                    "url",
                    f"https://news.ycombinator.com/item?id={story_id}",
                )
                score = item.get("score", 0)

                stories.append(
                    f"{len(stories) + 1}. {title} ({score} points)\n   URL: {url}"
                )

                if len(stories) == limit:
                    break

            if not stories:
                return "No stories retrieved."

            return "\n\n".join(stories)

        except Exception as e:
            return f"Error fetching Hacker News top stories: {str(e)}"
