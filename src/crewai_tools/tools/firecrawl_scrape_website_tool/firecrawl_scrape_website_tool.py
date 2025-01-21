from typing import Any, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = Any


class FirecrawlScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    timeout: Optional[int] = Field(
        default=30000,
        description="Timeout in milliseconds for the scraping operation. The default value is 30000.",
    )


class FirecrawlScrapeWebsiteTool(BaseTool):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl web scrape tool"
    description: str = "Scrape webpages using Firecrawl and return the contents"
    args_schema: Type[BaseModel] = FirecrawlScrapeWebsiteToolSchema
    api_key: Optional[str] = None
    _firecrawl: Optional["FirecrawlApp"] = PrivateAttr(None)

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

    def _run(
        self,
        url: str,
        timeout: Optional[int] = 30000,
    ):
        options = {
            "formats": ["markdown"],
            "onlyMainContent": True,
            "includeTags": [],
            "excludeTags": [],
            "headers": {},
            "waitFor": 0,
            "timeout": timeout,
        }
        return self._firecrawl.scrape_url(url, options)


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
