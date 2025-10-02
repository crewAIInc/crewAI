import os
from typing import Any
from urllib.parse import urlencode

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class SerplyNewsSearchToolSchema(BaseModel):
    """Input for Serply News Search."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to fetch news articles"
    )


class SerplyNewsSearchTool(BaseTool):
    name: str = "News Search"
    description: str = "A tool to perform News article search with a search_query."
    args_schema: type[BaseModel] = SerplyNewsSearchToolSchema
    search_url: str = "https://api.serply.io/v1/news/"
    proxy_location: str | None = "US"
    headers: dict | None = Field(default_factory=dict)
    limit: int | None = 10
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SERPLY_API_KEY",
                description="API key for Serply services",
                required=True,
            ),
        ]
    )

    def __init__(
        self, limit: int | None = 10, proxy_location: str | None = "US", **kwargs
    ):
        """param: limit (int): The maximum number of results to return [10-100, defaults to 10]
        proxy_location: (str): Where to get news, specifically for a specific country results.
             ['US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'] (defaults to US).
        """
        super().__init__(**kwargs)
        self.limit = limit
        self.proxy_location = proxy_location
        self.headers = {
            "X-API-KEY": os.environ["SERPLY_API_KEY"],
            "User-Agent": "crew-tools",
            "X-Proxy-Location": proxy_location,
        }

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        # build query parameters
        query_payload = {}

        if "query" in kwargs:
            query_payload["q"] = kwargs["query"]
        elif "search_query" in kwargs:
            query_payload["q"] = kwargs["search_query"]

        # build the url
        url = f"{self.search_url}{urlencode(query_payload)}"

        response = requests.request(
            "GET",
            url,
            headers=self.headers,
            timeout=30,
        )
        results = response.json()
        if "entries" in results:
            results = results["entries"]
            string = []
            for result in results[: self.limit]:
                try:
                    # follow url
                    r = requests.get(
                        result["link"],
                        timeout=30,
                    )
                    final_link = r.history[-1].headers["Location"]
                    string.append(
                        "\n".join(
                            [
                                f"Title: {result['title']}",
                                f"Link: {final_link}",
                                f"Source: {result['source']['title']}",
                                f"Published: {result['published']}",
                                "---",
                            ]
                        )
                    )
                except KeyError:  # noqa: PERF203
                    continue

            content = "\n".join(string)
            return f"\nSearch results: {content}\n"
        return results
