from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl web search tool"
    description: str = "Search webpages using Firecrawl and return the results"
    args_schema: Type[BaseModel] = FirecrawlSearchToolSchema
    api_key: Optional[str] = None
    _firecrawl: Optional["FirecrawlApp"] = PrivateAttr(None)

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from firecrawl import FirecrawlApp  # type: ignore
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )
        self._firecrawl = FirecrawlApp(api_key=api_key)

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
            "limit": limit,
            "tbs": tbs,
            "lang": lang,
            "country": country,
            "location": location,
            "timeout": timeout,
            "scrapeOptions": scrape_options,
        }
        return self._firecrawl.search(query, options)


try:
    from firecrawl import FirecrawlApp

    # Rebuild the model after class is defined
    FirecrawlSearchTool.model_rebuild()
except ImportError:
    # Exception can be ignored if the tool is not used
    pass
