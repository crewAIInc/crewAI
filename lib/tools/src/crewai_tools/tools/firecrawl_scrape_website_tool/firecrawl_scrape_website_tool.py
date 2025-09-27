from typing import Any, Optional, Type, Dict, List, TYPE_CHECKING

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from firecrawl import FirecrawlApp

try:
    from firecrawl import FirecrawlApp

    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False


class FirecrawlScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")


class FirecrawlScrapeWebsiteTool(BaseTool):
    """
    Tool for scraping webpages using Firecrawl. To run this tool, you need to have a Firecrawl API key.

    Args:
        api_key (str): Your Firecrawl API key.
        config (dict): Optional. It contains Firecrawl API parameters.

    Default configuration options:
        formats (list[str]): Content formats to return. Default: ["markdown"]
        onlyMainContent (bool): Only return main content. Default: True
        includeTags (list[str]): Tags to include. Default: []
        excludeTags (list[str]): Tags to exclude. Default: []
        headers (dict): Headers to include. Default: {}
        waitFor (int): Time to wait for page to load in ms. Default: 0
        json_options (dict): Options for JSON extraction. Default: None
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl web scrape tool"
    description: str = "Scrape webpages using Firecrawl and return the contents"
    args_schema: Type[BaseModel] = FirecrawlScrapeWebsiteToolSchema
    api_key: Optional[str] = None
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "formats": ["markdown"],
            "onlyMainContent": True,
            "includeTags": [],
            "excludeTags": [],
            "headers": {},
            "waitFor": 0,
        }
    )

    _firecrawl: Optional["FirecrawlApp"] = PrivateAttr(None)
    package_dependencies: List[str] = ["firecrawl-py"]
    env_vars: List[EnvVar] = [
        EnvVar(name="FIRECRAWL_API_KEY", description="API key for Firecrawl services", required=True),
    ]

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from firecrawl import FirecrawlApp  # type: ignore
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'firecrawl-py' package. Would you like to install it?"
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

        self._firecrawl = FirecrawlApp(api_key=api_key)

    def _run(self, url: str):
        if not self._firecrawl:
            raise RuntimeError("FirecrawlApp not properly initialized")

        return self._firecrawl.scrape_url(url, params=self.config)


try:
    from firecrawl import FirecrawlApp

    # Must rebuild model after class is defined
    if not hasattr(FirecrawlScrapeWebsiteTool, "_model_rebuilt"):
        FirecrawlScrapeWebsiteTool.model_rebuild()
        FirecrawlScrapeWebsiteTool._model_rebuilt = True
except ImportError:
    """
    When this tool is not used, then exception can be ignored.
    """
