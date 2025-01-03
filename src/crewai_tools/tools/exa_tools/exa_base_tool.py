from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class EXABaseToolToolSchema(BaseModel):
    """Input for EXABaseTool."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )


class EXABaseTool(BaseTool):
    name: str = "Search the internet"
    description: str = (
        "A tool that can be used to search the internet from a search_query"
    )
    args_schema: Type[BaseModel] = EXABaseToolToolSchema
    search_url: str = "https://api.exa.ai/search"
    n_results: int = None
    headers: dict = {
        "accept": "application/json",
        "content-type": "application/json",
    }

    def _parse_results(self, results):
        string = []
        for result in results:
            try:
                string.append(
                    "\n".join(
                        [
                            f"Title: {result['title']}",
                            f"Score: {result['score']}",
                            f"Url: {result['url']}",
                            f"ID: {result['id']}",
                            "---",
                        ]
                    )
                )
            except KeyError:
                continue

        content = "\n".join(string)
        return f"\nSearch results: {content}\n"
