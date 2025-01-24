from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from firecrawl import FirecrawlApp


try:
    from firecrawl import FirecrawlApp

    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False


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
        self.api_key = api_key
        self._initialize_firecrawl()

    def _initialize_firecrawl(self) -> None:
        try:
            if FIRECRAWL_AVAILABLE:
                self._firecrawl = FirecrawlApp(api_key=self.api_key)
            else:
                raise ImportError
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'firecrawl-py' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "firecrawl-py"], check=True)
                    from firecrawl import FirecrawlApp

                    self.firecrawl = FirecrawlApp(api_key=self.api_key)
                except subprocess.CalledProcessError:
                    raise ImportError("Failed to install firecrawl-py package")
            else:
                raise ImportError(
                    "`firecrawl-py` package not found, please run `uv add firecrawl-py`"
                )

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
    ) -> Any:
        if not self.firecrawl:
            raise RuntimeError("FirecrawlApp not properly initialized")

        options = {
            "limit": limit,
            "tbs": tbs,
            "lang": lang,
            "country": country,
            "location": location,
            "timeout": timeout,
            "scrapeOptions": scrape_options or {},
        }
        return self.firecrawl.search(**options)


try:
    from firecrawl import FirecrawlApp  # type: ignore

    # Only rebuild if the class hasn't been initialized yet
    if not hasattr(FirecrawlSearchTool, "_model_rebuilt"):
        FirecrawlSearchTool.model_rebuild()
        FirecrawlSearchTool._model_rebuilt = True
except ImportError:
    """
    When this tool is not used, then exception can be ignored.
    """
    pass
