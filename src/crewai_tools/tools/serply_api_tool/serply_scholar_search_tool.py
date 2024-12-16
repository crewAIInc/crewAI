import os
from typing import Any, Optional, Type
from urllib.parse import urlencode

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


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
    args_schema: Type[BaseModel] = SerplyScholarSearchToolSchema
    search_url: str = "https://api.serply.io/v1/scholar/"
    hl: Optional[str] = "us"
    proxy_location: Optional[str] = "US"
    headers: Optional[dict] = {}

    def __init__(self, hl: str = "us", proxy_location: Optional[str] = "US", **kwargs):
        """
        param: hl (str): host Language code to display results in
            (reference https://developers.google.com/custom-search/docs/xml_results?hl=en#wsInterfaceLanguages)
        proxy_location: (str): Specify the proxy location for the search, specifically for a specific country results.
             ['US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'] (defaults to US)
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

        response = requests.request("GET", url, headers=self.headers)
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
            except KeyError:
                continue

        content = "\n".join(string)
        return f"\nSearch results: {content}\n"
