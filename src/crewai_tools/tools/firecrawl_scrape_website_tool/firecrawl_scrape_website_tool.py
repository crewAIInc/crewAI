from typing import TYPE_CHECKING, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

# Type checking import
if TYPE_CHECKING:
    from firecrawl import FirecrawlApp


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
    description: str = "Scrape webpages url using Firecrawl and return the contents"
    args_schema: Type[BaseModel] = FirecrawlScrapeWebsiteToolSchema
    api_key: Optional[str] = None
    firecrawl: Optional["FirecrawlApp"] = None  # Updated to use TYPE_CHECKING

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
        return self.firecrawl.scrape_url(url, options)


try:
    from firecrawl import FirecrawlApp

    # Must rebuild model after class is defined
    FirecrawlScrapeWebsiteTool.model_rebuild()
except ImportError:
    """
    When this tool is not used, then exception can be ignored.
    """
    pass
