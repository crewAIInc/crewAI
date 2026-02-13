import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


# Country codes supported by You.com Search API
Country = Literal[
    "AR",
    "AU",
    "AT",
    "BE",
    "BR",
    "CA",
    "CL",
    "DK",
    "FI",
    "FR",
    "DE",
    "HK",
    "IN",
    "ID",
    "IT",
    "JP",
    "KR",
    "MY",
    "MX",
    "NL",
    "NZ",
    "NO",
    "CN",
    "PL",
    "PT",
    "PT-BR",
    "PH",
    "RU",
    "SA",
    "ZA",
    "ES",
    "SE",
    "CH",
    "TW",
    "TR",
    "GB",
    "US",
]

# Language codes supported by You.com Search API (BCP 47 format)
Language = Literal[
    "AR",
    "EU",
    "BN",
    "BG",
    "CA",
    "ZH-HANS",
    "ZH-HANT",
    "HR",
    "CS",
    "DA",
    "NL",
    "EN",
    "EN-GB",
    "EN-US",
    "ET",
    "FI",
    "FR",
    "GL",
    "DE",
    "EL",
    "GU",
    "HE",
    "HI",
    "HU",
    "IS",
    "IT",
    "JP",
    "KN",
    "KO",
    "LV",
    "LT",
    "MS",
    "ML",
    "MR",
    "NB",
    "PL",
    "PT-BR",
    "PT-PT",
    "PA",
    "RO",
    "RU",
    "SR",
    "SK",
    "SL",
    "ES",
    "SV",
    "TA",
    "TE",
    "TH",
    "TR",
    "UK",
    "VI",
]


class YouSearchToolSchema(BaseModel):
    """Input for YouSearchTool."""

    query: str = Field(..., description="Search query to perform")


class YouSearchTool(BaseTool):
    """A tool that performs web searches using the You.com Search API."""

    name: str = "You.com Search"
    description: str = (
        "A tool that performs web searches using the You.com Search API. "
        "It returns a JSON object containing the search results with support for "
        "advanced search operators and filters."
    )
    args_schema: type[BaseModel] = YouSearchToolSchema
    search_url: str = "https://api.ydc-index.io/search"
    count: int = 10
    offset: int | None = None
    country: Country | None = "US"
    language: Language | None = None
    freshness: str | None = None
    safesearch: Literal["off", "moderate", "strict"] = "moderate"
    livecrawl: Literal["web", "news", "all"] | None = None
    livecrawl_formats: Literal["html", "markdown"] | None = "markdown"
    timeout: int = 60
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="YOU_API_KEY",
                description="API key for You.com search service",
                required=True,
            ),
        ],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "YOU_API_KEY" not in os.environ:
            raise ValueError(
                "YOU_API_KEY environment variable is required for YouSearchTool. "
                "Get your API key at https://you.com/platform/api-keys",
            )

    def _run(
        self,
        query: str,
    ) -> str:
        """Execute the search operation.

        Args:
            query: The search query string.

        Returns:
            JSON string containing the search results.
        """
        try:
            if not query:
                raise ValueError("Query is required")

            # Build request parameters
            params: dict[str, Any] = {
                "query": query,
                "count": max(1, min(self.count, 100)),
            }

            # Add offset if specified (range: 0-9)
            if self.offset is not None:
                params["offset"] = max(0, min(self.offset, 9))

            if self.country:
                params["country"] = self.country

            if self.language:
                params["language"] = self.language

            if self.freshness:
                params["freshness"] = self.freshness

            if self.safesearch:
                params["safesearch"] = self.safesearch

            if self.livecrawl:
                params["livecrawl"] = self.livecrawl

            if self.livecrawl_formats:
                params["livecrawl_formats"] = self.livecrawl_formats

            # Setup request headers
            headers = {
                "X-API-Key": os.environ["YOU_API_KEY"],
                "Accept": "application/json",
            }

            # Make API request
            response = requests.get(
                self.search_url,
                headers=headers,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json()

            return json.dumps(results, indent=2)

        except requests.RequestException as e:
            return f"Error performing search: {e!s}"
        except KeyError as e:
            return f"Error parsing search results: {e!s}"
        except ValueError as e:
            return f"Invalid parameters: {e!s}"
