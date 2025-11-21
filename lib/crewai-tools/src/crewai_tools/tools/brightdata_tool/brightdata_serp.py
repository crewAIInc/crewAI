import os
from typing import Any
import urllib.parse

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


class BrightDataConfig(BaseModel):
    API_URL: str = "https://api.brightdata.com/request"

    @classmethod
    def from_env(cls):
        return cls(
            API_URL=os.environ.get(
                "BRIGHTDATA_API_URL", "https://api.brightdata.com/request"
            )
        )


class BrightDataSearchToolSchema(BaseModel):
    """Schema that defines the input arguments for the BrightDataSearchToolSchema.

    Attributes:
        query (str): The search query to be executed (e.g., "latest AI news").
        search_engine (Optional[str]): The search engine to use ("google", "bing", "yandex"). Default is "google".
        country (Optional[str]): Two-letter country code for geo-targeting (e.g., "us", "in"). Default is "us".
        language (Optional[str]): Language code for search results (e.g., "en", "es"). Default is "en".
        search_type (Optional[str]): Type of search, such as "isch" (images), "nws" (news), "jobs", etc.
        device_type (Optional[str]): Device type to simulate ("desktop", "mobile", "ios", "android"). Default is "desktop".
        parse_results (Optional[bool]): If True, results will be returned in structured JSON. If False, raw HTML. Default is True.
    """

    query: str = Field(..., description="Search query to perform")
    search_engine: str | None = Field(
        default="google",
        description="Search engine domain (e.g., 'google', 'bing', 'yandex')",
    )
    country: str | None = Field(
        default="us",
        description="Two-letter country code for geo-targeting (e.g., 'us', 'gb')",
    )
    language: str | None = Field(
        default="en",
        description="Language code (e.g., 'en', 'es') used in the query URL",
    )
    search_type: str | None = Field(
        default=None,
        description="Type of search (e.g., 'isch' for images, 'nws' for news)",
    )
    device_type: str | None = Field(
        default="desktop",
        description="Device type to simulate (e.g., 'mobile', 'desktop', 'ios')",
    )
    parse_results: bool | None = Field(
        default=True,
        description="Whether to parse and return JSON (True) or raw HTML/text (False)",
    )


class BrightDataSearchTool(BaseTool):
    """A web search tool that utilizes Bright Data's SERP API to perform queries and return either structured results
    or raw page content from search engines like Google or Bing.

    Attributes:
        name (str): Tool name used by the agent.
        description (str): A brief explanation of what the tool does.
        args_schema (Type[BaseModel]): Schema class for validating tool arguments.
        base_url (str): The Bright Data API endpoint used for making the POST request.
        api_key (str): Bright Data API key loaded from the environment variable 'BRIGHT_DATA_API_KEY'.
        zone (str): Zone identifier from Bright Data, loaded from the environment variable 'BRIGHT_DATA_ZONE'.

    Raises:
        ValueError: If API key or zone environment variables are not set.
    """

    name: str = "Bright Data SERP Search"
    description: str = "Tool to perform web search using Bright Data SERP API."
    args_schema: type[BaseModel] = BrightDataSearchToolSchema
    _config = BrightDataConfig.from_env()
    base_url: str = ""
    api_key: str = ""
    zone: str = ""
    query: str | None = None
    search_engine: str = "google"
    country: str = "us"
    language: str = "en"
    search_type: str | None = None
    device_type: str = "desktop"
    parse_results: bool = True
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BRIGHT_DATA_API_KEY",
                description="API key for Bright Data",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        query: str | None = None,
        search_engine: str = "google",
        country: str = "us",
        language: str = "en",
        search_type: str | None = None,
        device_type: str = "desktop",
        parse_results: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.base_url = self._config.API_URL
        self.query = query
        self.search_engine = search_engine
        self.country = country
        self.language = language
        self.search_type = search_type
        self.device_type = device_type
        self.parse_results = parse_results

        self.api_key = os.getenv("BRIGHT_DATA_API_KEY") or ""
        self.zone = os.getenv("BRIGHT_DATA_ZONE") or ""
        if not self.api_key:
            raise ValueError("BRIGHT_DATA_API_KEY environment variable is required.")
        if not self.zone:
            raise ValueError("BRIGHT_DATA_ZONE environment variable is required.")

    def get_search_url(self, engine: str, query: str):
        if engine == "yandex":
            return f"https://yandex.com/search/?text=${query}"
        if engine == "bing":
            return f"https://www.bing.com/search?q=${query}"
        return f"https://www.google.com/search?q=${query}"

    def _run(
        self,
        query: str | None = None,
        search_engine: str | None = None,
        country: str | None = None,
        language: str | None = None,
        search_type: str | None = None,
        device_type: str | None = None,
        parse_results: bool | None = None,
        **kwargs,
    ) -> Any:
        """Executes a search query using Bright Data SERP API and returns results.

        Args:
            query (str): The search query string (URL encoded internally).
            search_engine (str): The search engine to use (default: "google").
            country (str): Country code for geotargeting (default: "us").
            language (str): Language code for the query (default: "en").
            search_type (str): Optional type of search such as "nws", "isch", "jobs".
            device_type (str): Optional device type to simulate (e.g., "mobile", "ios", "desktop").
            parse_results (bool): If True, returns structured data; else raw page (default: True).
            results_count (str or int): Number of search results to fetch (default: "10").

        Returns:
            dict or str: Parsed JSON data from Bright Data if available, otherwise error message.
        """
        query = query or self.query
        search_engine = search_engine or self.search_engine
        country = country or self.country
        language = language or self.language
        search_type = search_type or self.search_type
        device_type = device_type or self.device_type
        parse_results = (
            parse_results if parse_results is not None else self.parse_results
        )
        results_count = kwargs.get("results_count", "10")

        # Validate required parameters
        if not query:
            raise ValueError("query is required either in constructor or method call")

        # Build the search URL
        query = urllib.parse.quote(query)
        url = self.get_search_url(search_engine, query)

        # Add parameters to the URL
        params = []

        if country:
            params.append(f"gl={country}")

        if language:
            params.append(f"hl={language}")

        if results_count:
            params.append(f"num={results_count}")

        if parse_results:
            params.append("brd_json=1")

        if search_type:
            if search_type == "jobs":
                params.append("ibp=htl;jobs")
            else:
                params.append(f"tbm={search_type}")

        if device_type:
            if device_type == "mobile":
                params.append("brd_mobile=1")
            elif device_type == "ios":
                params.append("brd_mobile=ios")
            elif device_type == "android":
                params.append("brd_mobile=android")

        # Combine parameters with the URL
        if params:
            url += "&" + "&".join(params)

        # Set up the API request parameters
        request_params = {"zone": self.zone, "url": url, "format": "raw"}

        request_params = {k: v for k, v in request_params.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                self.base_url, json=request_params, headers=headers, timeout=30
            )

            response.raise_for_status()

            return response.text

        except requests.RequestException as e:
            return f"Error performing BrightData search: {e!s}"
        except Exception as e:
            return f"Error fetching results: {e!s}"
