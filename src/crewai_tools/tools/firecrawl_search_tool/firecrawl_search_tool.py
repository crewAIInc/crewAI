from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

# Type checking import
if TYPE_CHECKING:
    from firecrawl import FirecrawlApp


class FirecrawlSearchToolSchema(BaseModel):
    query: str = Field(description="Search query")
    limit: Optional[int] = Field(
        default=5, description="Maximum number of results to return"
    )
    tbs: Optional[str] = Field(default=None, description="Time-based search parameter")
    lang: Optional[str] = Field(
        default="en", description="Language code for search results"
    )
    country: Optional[str] = Field(
        default="us", description="Country code for search results"
    )
    location: Optional[str] = Field(
        default=None, description="Location parameter for search results"
    )
    timeout: Optional[int] = Field(default=60000, description="Timeout in milliseconds")
    scrape_options: Optional[Dict[str, Any]] = Field(
        default=None, description="Options for scraping search results"
    )


class FirecrawlSearchTool(BaseTool):
    name: str = "Firecrawl web search tool"
    description: str = "Search webpages using Firecrawl and return the results"
    args_schema: Type[BaseModel] = FirecrawlSearchToolSchema
    api_key: Optional[str] = None
    firecrawl: Optional["FirecrawlApp"] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from firecrawl import FirecrawlApp  # type: ignore
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'firecrawl-py' package. Would you like to install it? (y/N)"
            ):
                import subprocess

                subprocess.run(["uv", "add", "firecrawl-py"], check=True)
                from firecrawl import (
                    FirecrawlApp,
                )
            else:
                raise ImportError(
                    "`firecrawl-py` package not found, please run `uv add firecrawl-py`"
                )

        self.firecrawl = FirecrawlApp(api_key=api_key)

    def _run(
        self,
        query: str,
        limit: Optional[int] = 5,
        tbs: Optional[str] = None,
        lang: Optional[str] = "en",
        country: Optional[str] = "us",
        location: Optional[str] = None,
        timeout: Optional[int] = 60000,
        scrape_options: Optional[Dict[str, Any]] = None,
    ):
        if scrape_options is None:
            scrape_options = {}

        options = {
            "query": query,
            "limit": limit,
            "tbs": tbs,
            "lang": lang,
            "country": country,
            "location": location,
            "timeout": timeout,
            "scrapeOptions": scrape_options,
        }
        return self.firecrawl.search(**options)
