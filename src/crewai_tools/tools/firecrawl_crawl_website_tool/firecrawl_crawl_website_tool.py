from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

try:
    from firecrawl import FirecrawlApp, ScrapeOptions
except ImportError:
    FirecrawlApp = Any


class FirecrawlCrawlWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")


class FirecrawlCrawlWebsiteTool(BaseTool):
    """
    Tool for crawling websites using Firecrawl. To run this tool, you need to have a Firecrawl API key.

    Args:
        api_key (str): Your Firecrawl API key.
        config (dict): Optional. It contains Firecrawl API parameters.

    Default configuration options:
        max_depth (int): Maximum depth to crawl. Default: 2
        ignore_sitemap (bool): Whether to ignore sitemap. Default: True
        limit (int): Maximum number of pages to crawl. Default: 100
        allow_backward_links (bool): Allow crawling backward links. Default: False
        allow_external_links (bool): Allow crawling external links. Default: False
        scrape_options (ScrapeOptions): Options for scraping content
            - formats (list[str]): Content formats to return. Default: ["markdown", "screenshot", "links"]
            - only_main_content (bool): Only return main content. Default: True
            - timeout (int): Timeout in milliseconds. Default: 30000
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl web crawl tool"
    description: str = "Crawl webpages using Firecrawl and return the contents"
    args_schema: Type[BaseModel] = FirecrawlCrawlWebsiteToolSchema
    api_key: Optional[str] = None
    config: Optional[dict[str, Any]] = Field(
        default_factory=lambda: {
            "max_depth": 2,
            "ignore_sitemap": True,
            "limit": 100,
            "allow_backward_links": False,
            "allow_external_links": False,
            "scrape_options": ScrapeOptions(
                formats=["markdown", "screenshot", "links"],
                only_main_content=True,
                timeout=30000,
            ),
        }
    )
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

    def _run(self, url: str):
        return self._firecrawl.crawl_url(url, **self.config)


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
