from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = Any


class FirecrawlCrawlWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    maxDepth: Optional[int] = Field(
        default=2,
        description="Maximum depth to crawl. Depth 1 is the base URL, depth 2 includes the base URL and its direct children and so on.",
    )
    limit: Optional[int] = Field(
        default=100, description="Maximum number of pages to crawl."
    )
    allowExternalLinks: Optional[bool] = Field(
        default=False,
        description="Allows the crawler to follow links that point to external domains.",
    )
    formats: Optional[list[str]] = Field(
        default=["markdown", "screenshot", "links"],
        description="Formats for the page's content to be returned (eg. markdown, html, screenshot, links).",
    )
    timeout: Optional[int] = Field(
        default=30000,
        description="Timeout in milliseconds for the crawling operation. The default value is 30000.",
    )


class FirecrawlCrawlWebsiteTool(BaseTool):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl web crawl tool"
    description: str = "Crawl webpages using Firecrawl and return the contents"
    args_schema: Type[BaseModel] = FirecrawlCrawlWebsiteToolSchema
    api_key: Optional[str] = None
    _firecrawl: Optional["FirecrawlApp"] = PrivateAttr(None)

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self._initialize_firecrawl()

    def _initialize_firecrawl(self) -> None:
        try:
            from firecrawl import FirecrawlApp  # type: ignore

            self._firecrawl = FirecrawlApp(api_key=self.api_key)
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'firecrawl-py' package. Would you like to install it?"
            ):
                import subprocess

                try:
                    subprocess.run(["uv", "add", "firecrawl-py"], check=True)
                    from firecrawl import FirecrawlApp

                    self._firecrawl = FirecrawlApp(api_key=self.api_key)
                except subprocess.CalledProcessError:
                    raise ImportError("Failed to install firecrawl-py package")
            else:
                raise ImportError(
                    "`firecrawl-py` package not found, please run `uv add firecrawl-py`"
                )

    def _run(
        self,
        url: str,
        maxDepth: Optional[int] = 2,
        limit: Optional[int] = 100,
        allowExternalLinks: Optional[bool] = False,
        formats: Optional[list[str]] = ["markdown", "screenshot", "links"],
        timeout: Optional[int] = 30000,
    ):
        # Default options for timeout and crawling
        DEFAULT_TIMEOUT = 30000
        DEFAULT_CRAWLING_OPTIONS = {
            "maxDepth": 2,
            "ignoreSitemap": True,
            "limit": 100,
            "allowBackwardLinks": False,
            "allowExternalLinks": False,
            "scrapeOptions": {
                "formats": ["markdown", "screenshot", "links"],
                "onlyMainContent": True,
                "timeout": DEFAULT_TIMEOUT,
            },
        }

        # Add default options not present as parameters
        crawling_options = DEFAULT_CRAWLING_OPTIONS

        # Update the values of parameters present
        crawling_options["maxDepth"] = maxDepth
        crawling_options["limit"] = limit
        crawling_options["allowExternalLinks"] = allowExternalLinks
        crawling_options["scrapeOptions"]["formats"] = formats
        crawling_options["scrapeOptions"]["timeout"] = timeout

        return self._firecrawl.crawl_url(url, crawling_options)


try:
    from firecrawl import FirecrawlApp

    # Only rebuild if the class hasn't been initialized yet
    if not hasattr(FirecrawlCrawlWebsiteTool, "_model_rebuilt"):
        FirecrawlCrawlWebsiteTool.model_rebuild()
        FirecrawlCrawlWebsiteTool._model_rebuilt = True
except ImportError:
    """
    When this tool is not used, then exception can be ignored.
    """
