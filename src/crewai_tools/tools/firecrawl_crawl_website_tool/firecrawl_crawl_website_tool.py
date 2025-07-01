from typing import Any, Optional, Type, List, TYPE_CHECKING

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from firecrawl import FirecrawlApp

try:
    from firecrawl import FirecrawlApp

    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False


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
            "maxDepth": 2,
            "ignoreSitemap": True,
            "limit": 10,
            "allowBackwardLinks": False,
            "allowExternalLinks": False,
            "scrapeOptions": {
                "formats": ["markdown", "screenshot", "links"],
                "onlyMainContent": True,
                "timeout": 10000,
            },
        }
    )
    _firecrawl: Optional["FirecrawlApp"] = PrivateAttr(None)
    package_dependencies: List[str] = ["firecrawl-py"]
    env_vars: List[EnvVar] = [
        EnvVar(name="FIRECRAWL_API_KEY", description="API key for Firecrawl services", required=True),
    ]

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
        if not self._firecrawl:
            raise RuntimeError("FirecrawlApp not properly initialized")

        return self._firecrawl.crawl_url(url, poll_interval=2, params=self.config)


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
