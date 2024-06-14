import os
import requests
from urllib.parse import urlencode
from typing import Type, Any, Optional
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool


class SerplyWebSearchToolSchema(BaseModel):
    """Input for Serply Web Search."""
    search_query: str = Field(..., description="Mandatory search query you want to use to Google search")


class SerplyWebSearchTool(BaseTool):
    name: str = "Google Search"
    description: str = "A tool to perform Google search with a search_query."
    args_schema: Type[BaseModel] = SerplyWebSearchToolSchema
    search_url: str = "https://api.serply.io/v1/search/"
    hl: Optional[str] = "us"
    limit: Optional[int] = 10
    device_type: Optional[str] = "desktop"
    proxy_location: Optional[str] = "US"
    query_payload: Optional[dict] = {}
    headers: Optional[dict] = {}

    def __init__(
            self,
            hl: str = "us",
            limit: int = 10,
            device_type: str = "desktop",
            proxy_location: str = "US",
            **kwargs
    ):
        """
            param: query (str): The query to search for
            param: hl (str): host Language code to display results in
                (reference https://developers.google.com/custom-search/docs/xml_results?hl=en#wsInterfaceLanguages)
            param: limit (int): The maximum number of results to return [10-100, defaults to 10]
            param: device_type (str): desktop/mobile results (defaults to desktop)
            proxy_location: (str): Where to perform the search, specifically for local/regional results.
                 ['US', 'CA', 'IE', 'GB', 'FR', 'DE', 'SE', 'IN', 'JP', 'KR', 'SG', 'AU', 'BR'] (defaults to US)
        """
        super().__init__(**kwargs)

        self.limit = limit
        self.device_type = device_type
        self.proxy_location = proxy_location

        # build query parameters
        self.query_payload = {
            "num": limit,
            "gl": proxy_location.upper(),
            "hl": hl.lower()
        }
        self.headers = {
            "X-API-KEY": os.environ["SERPLY_API_KEY"],
            "X-User-Agent": device_type,
            "User-Agent": "crew-tools",
            "X-Proxy-Location": proxy_location
        }

    def _run(
            self,
            **kwargs: Any,
    ) -> Any:
        if "query" in kwargs:
            self.query_payload["q"] = kwargs["query"]
        elif "search_query" in kwargs:
            self.query_payload["q"] = kwargs["search_query"]

        # build the url
        url = f"{self.search_url}{urlencode(self.query_payload)}"

        response = requests.request("GET", url, headers=self.headers)
        results = response.json()
        if "results" in results:
            results = results['results']
            string = []
            for result in results:
                try:
                    string.append('\n'.join([
                        f"Title: {result['title']}",
                        f"Link: {result['link']}",
                        f"Description: {result['description'].strip()}",
                        "---"
                    ]))
                except KeyError:
                    continue

            content = '\n'.join(string)
            return f"\nSearch results: {content}\n"
        else:
            return results
