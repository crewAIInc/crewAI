from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from crewai_tools.tools.base_tool import BaseTool


class FirecrawlCrawlWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    crawler_options: Optional[Dict[str, Any]] = Field(
        default=None, description="Options for crawling"
    )
    page_options: Optional[Dict[str, Any]] = Field(
        default=None, description="Options for page"
    )


class FirecrawlCrawlWebsiteTool(BaseTool):
    name: str = "Firecrawl web crawl tool"
    description: str = "Crawl webpages using Firecrawl and return the contents"
    args_schema: Type[BaseModel] = FirecrawlCrawlWebsiteToolSchema
    api_key: Optional[str] = None
    firecrawl: Optional[Any] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from firecrawl import FirecrawlApp  # type: ignore
        except ImportError:
            raise ImportError(
                "`firecrawl` package not found, please run `pip install firecrawl-py`"
            )

        self.firecrawl = FirecrawlApp(api_key=api_key)

    def _run(
        self,
        url: str,
        crawler_options: Optional[Dict[str, Any]] = None,
        page_options: Optional[Dict[str, Any]] = None,
    ):
        if crawler_options is None:
            crawler_options = {}
        if page_options is None:
            page_options = {}

        options = {"crawlerOptions": crawler_options, "pageOptions": page_options}
        return self.firecrawl.crawl_url(url, options)
