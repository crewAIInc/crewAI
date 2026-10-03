from typing import Type
import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class HackerNewsInput(BaseModel):
    """Input schema for HackerNewsTopStoriesTool."""
    limit: int = Field(
        default=5,
        description="Number of top stories to fetch from Hacker News (max 20)."
    )

class HackerNewsTopStoriesTool(BaseTool):
    name: str = "Hacker News Top Stories"
    description: str = (
        "Fetches current top stories from Hacker News including title, score, and URL. "
        "Useful for agents analyzing tech news or identifying trending tech topics."
    )
    args_schema: Type[BaseModel] = HackerNewsInput

    def _run(self, limit: int = 5) -> str:
        try:
            count = min(max(1, limit), 20)
            res = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10)
            res.raise_for_status()
            story_ids = res.json()[:count]

            stories = []
            for idx, story_id in enumerate(story_ids, start=1):
                item_res = requests.get(
                    f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                    timeout=5
                )
                if item_res.status_code == 200:
                    item = item_res.json()
                    title = item.get("title", "Untitled")
                    url = item.get("url", f"https://news.ycombinator.com/item?id={story_id}")
                    score = item.get("score", 0)
                    stories.append(f"{idx}. {title} ({score} points)\n   URL: {url}")

            return "\n\n".join(stories) if stories else "No stories retrieved."
        except Exception as err:
            return f"Failed to fetch Hacker News stories: {str(err)}"
