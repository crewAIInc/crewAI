from typing import Any, Dict, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = Any


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
        crawler_options: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = 30000,
    ):
        if crawler_options is None:
            crawler_options = {}

        options = {
            "crawlerOptions": crawler_options,
            "timeout": timeout,
        }
        return self._firecrawl.crawl_url(url, options)


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
