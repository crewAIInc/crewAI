import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from pydantic import BaseModel, ConfigDict, Field

from crewai.tools import BaseTool

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
    firecrawl_app: Optional["FirecrawlApp"] = None
    api_key: Optional[str] = None
    url: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    poll_interval: Optional[int] = 2
    idempotency_key: Optional[str] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize FirecrawlCrawlWebsiteTool.

        Args:
            api_key (Optional[str]): Firecrawl API key. If not provided, will check FIRECRAWL_API_KEY env var.
            url (Optional[str]): Base URL to crawl. Can be overridden by the _run method.
            firecrawl_app (Optional[FirecrawlApp]): Previously created FirecrawlApp instance.
            params (Optional[Dict[str, Any]]): Additional parameters to pass to the FirecrawlApp.
            poll_interval (Optional[int]): Poll interval for the FirecrawlApp.
            idempotency_key (Optional[str]): Idempotency key for the FirecrawlApp.
            **kwargs: Additional arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        try:
            from firecrawl import FirecrawlApp  # type: ignore
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )

        # Allows passing a previously created FirecrawlApp instance
        # or builds a new one with the provided API key
        if not self.firecrawl_app:
            client_api_key = api_key or os.getenv("FIRECRAWL_API_KEY")    
            if not client_api_key:
                raise ValueError(
                    "FIRECRAWL_API_KEY is not set. Please provide it either via the constructor "
                    "with the `api_key` argument or by setting the FIRECRAWL_API_KEY environment variable."
                )
            self.firecrawl_app = FirecrawlApp(api_key=client_api_key)

    def _run(self, url: str):
        # Unless url has been previously set via constructor by the user,
        # use the url argument provided by the agent at runtime.
        base_url = self.url or url

        return self.firecrawl_app.crawl_url(
            base_url, 
            params=self.params, 
            poll_interval=self.poll_interval, 
            idempotency_key=self.idempotency_key
        )


try:
    from firecrawl import FirecrawlApp

    # Must rebuild model after class is defined
    FirecrawlCrawlWebsiteTool.model_rebuild()
except ImportError:
    """
    When this tool is not used, then exception can be ignored.
    """
    pass
