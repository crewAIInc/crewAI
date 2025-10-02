import os
from typing import Any
from urllib.parse import urlencode

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class SerplyScholarSearchToolSchema(BaseModel):
    """Input for Serply Scholar Search."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to fetch scholarly literature",
    )


class SerplyScholarSearchTool(BaseTool):
    name: str = "Scholar Search"
    description: str = (
        "A tool to perform scholarly literature search with a search_query."
    )
    args_schema: type[BaseModel] = SerplyScholarSearchToolSchema
    search_url: str = "https://api.serply.io/v1/scholar/"
    hl: str | None = "us"
    proxy_location: str | None = "US"
    headers: dict | None = Field(default_factory=dict)
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SERPLY_API_KEY",
                description="API key for Serply services",
                required=True,
            ),
        ]
    )

    def __init__(self, hl: str = "us", proxy_location: str | None = "US", **kwargs):
        """param: hl (str): host Language code to display results in
            (reference https://developers.google.com/custom-search/docs/xml_results?hl=en#wsInterfaceLanguages)
        proxy_location: (str): Specify the proxy location for the search, specifically for a specific country results.
             ['US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'] (defaults to US).
        """
        super().__init__(**kwargs)
        self.hl = hl
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
        query_payload = {"hl": self.hl}

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
        articles = response.json().get("articles", "")

        if not articles:
            return ""

        string = []
        for article in articles:
            try:
                if "doc" in article:
                    link = article["doc"]["link"]
                else:
                    link = article["link"]
                authors = [author["name"] for author in article["author"]["authors"]]
                string.append(
                    "\n".join(
                        [
                            f"Title: {article['title']}",
                            f"Link: {link}",
                            f"Description: {article['description']}",
                            f"Cite: {article['cite']}",
                            f"Authors: {', '.join(authors)}",
                            "---",
                        ]
                    )
                )
            except KeyError:  # noqa: PERF203
                continue

        content = "\n".join(string)
        return f"\nSearch results: {content}\n"
