import datetime
import os
from typing import Any, Optional, Type

import requests
from pydantic import BaseModel, Field

from crewai_tools.tools.base_tool import BaseTool


def _save_results_to_file(content: str) -> None:
    """Saves the search results to a file."""
    filename = f"search_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(filename, "w") as file:
        file.write(content)
    print(f"Results saved to {filename}")


class BraveSearchToolSchema(BaseModel):
    """Input for BraveSearchTool."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )


class BraveSearchTool(BaseTool):
    name: str = "Search the internet"
    description: str = (
        "A tool that can be used to search the internet with a search_query."
    )
    args_schema: Type[BaseModel] = BraveSearchToolSchema
    search_url: str = "https://api.search.brave.com/res/v1/web/search"
    country: Optional[str] = ""
    n_results: int = 10
    save_file: bool = False

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        search_query = kwargs.get("search_query") or kwargs.get("query")
        save_file = kwargs.get("save_file", self.save_file)
        n_results = kwargs.get("n_results", self.n_results)

        payload = {"q": search_query, "count": n_results}

        if self.country != "":
            payload["country"] = self.country

        headers = {
            "X-Subscription-Token": os.environ["BRAVE_API_KEY"],
            "Accept": "application/json",
        }

        response = requests.get(self.search_url, headers=headers, params=payload)
        results = response.json()

        if "web" in results:
            results = results["web"]["results"]
            string = []
            for result in results:
                try:
                    string.append(
                        "\n".join(
                            [
                                f"Title: {result['title']}",
                                f"Link: {result['url']}",
                                f"Snippet: {result['description']}",
                                "---",
                            ]
                        )
                    )
                except KeyError:
                    continue

            content = "\n".join(string)
            if save_file:
                _save_results_to_file(content)
            return f"\nSearch results: {content}\n"
        else:
            return results
