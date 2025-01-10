import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

# Type checking import
if TYPE_CHECKING:
    from firecrawl import FirecrawlApp


class FirecrawlCrawlWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    crawler_options: Optional[Dict[str, Any]] = Field(
        default=None, description="Options for crawling"
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

        if not self.firecrawl:
            client_api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
            if not client_api_key:
                raise ValueError(
                    "FIRECRAWL_API_KEY is not set. Please provide it either via the constructor "
                    "with the `api_key` argument or by setting the FIRECRAWL_API_KEY environment variable."
                )
            self.firecrawl = FirecrawlApp(api_key=client_api_key)

    def _run(
        self,
        url: str,
        crawler_options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = 30000,
    ):
        if crawler_options is None:
            crawler_options = {}

        options = {
            "crawlerOptions": crawler_options,
            "timeout": timeout,
        }
        return self.firecrawl.crawl_url(url, options)


try:
    from firecrawl import FirecrawlApp

    # Must rebuild model after class is defined
    FirecrawlCrawlWebsiteTool.model_rebuild()
except ImportError:
    """
    When this tool is not used, then exception can be ignored.
    """
    pass
