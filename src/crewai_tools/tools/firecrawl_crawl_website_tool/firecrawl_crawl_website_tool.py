from typing import TYPE_CHECKING, Any, Dict, Optional, Type
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
import os

# Type checking import
if TYPE_CHECKING:
    from firecrawl import FirecrawlApp


class FirecrawlCrawlWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")

class FirecrawlCrawlWebsiteTool(BaseTool):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Firecrawl web crawl tool"
    description: str = "Crawl webpages using Firecrawl and return the contents"
    args_schema: Type[BaseModel] = FirecrawlCrawlWebsiteToolSchema
    api_key: Optional[str] = None
    firecrawl: Optional["FirecrawlApp"] = None
    url: Optional[str] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from firecrawl import FirecrawlApp  # type: ignore
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )

        client_api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not client_api_key:
            raise ValueError("FIRECRAWL_API_KEY is not set")

        self.firecrawl = FirecrawlApp(api_key=client_api_key)

    def _run(
        self,
        url: str,
        crawler_options: Optional[Dict[str, Any]] = None,
        page_options: Optional[Dict[str, Any]] = None,
    ):
        # Unless url has been previously set via constructor by the user,
        # use the url argument provided by the agent
        base_url = self.url or url

        if crawler_options is None:
            crawler_options = {}
        if page_options is None:
            page_options = {}

        options = {"crawlerOptions": crawler_options, "pageOptions": page_options}
        return self.firecrawl.crawl_url(base_url, options)


try:
    from firecrawl import FirecrawlApp

    # Must rebuild model after class is defined
    FirecrawlCrawlWebsiteTool.model_rebuild()
except ImportError:
    """
    When this tool is not used, then exception can be ignored.
    """
    pass
