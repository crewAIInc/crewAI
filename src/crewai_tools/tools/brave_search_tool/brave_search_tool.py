import datetime
import os
import time
from typing import Any, ClassVar, Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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
    """
    BraveSearchTool - A tool for performing web searches using the Brave Search API.

    This module provides functionality to search the internet using Brave's Search API,
    supporting customizable result counts and country-specific searches.

    Dependencies:
        - requests
        - pydantic
        - python-dotenv (for API key management)
    """

    name: str = "Brave Web Search the internet"
    description: str = (
        "A tool that can be used to search the internet with a search_query."
    )
    args_schema: Type[BaseModel] = BraveSearchToolSchema
    search_url: str = "https://api.search.brave.com/res/v1/web/search"
    country: Optional[str] = ""
    n_results: int = 10
    save_file: bool = False
    _last_request_time: ClassVar[float] = 0
    _min_request_interval: ClassVar[float] = 1.0  # seconds

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "BRAVE_API_KEY" not in os.environ:
            raise ValueError(
                "BRAVE_API_KEY environment variable is required for BraveSearchTool"
            )

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        current_time = time.time()
        if (current_time - self._last_request_time) < self._min_request_interval:
            time.sleep(
                self._min_request_interval - (current_time - self._last_request_time)
            )
        BraveSearchTool._last_request_time = time.time()
        try:
            search_query = kwargs.get("search_query") or kwargs.get("query")
            if not search_query:
                raise ValueError("Search query is required")

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
            response.raise_for_status()  # Handle non-200 responses
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
        except requests.RequestException as e:
            return f"Error performing search: {str(e)}"
        except KeyError as e:
            return f"Error parsing search results: {str(e)}"
        if save_file:
            _save_results_to_file(content)
            return f"\nSearch results: {content}\n"
        else:
            return content
