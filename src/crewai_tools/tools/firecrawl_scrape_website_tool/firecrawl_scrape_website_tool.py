from typing import Optional, Any, Type, Dict
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

class FirecrawlScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    page_options: Optional[Dict[str, Any]] = Field(default=None, description="Options for page scraping")
    extractor_options: Optional[Dict[str, Any]] = Field(default=None, description="Options for data extraction")
    timeout: Optional[int] = Field(default=None, description="Timeout for the scraping operation")

class FirecrawlScrapeWebsiteTool(BaseTool):
    name: str = "Firecrawl web scrape tool"
    description: str = "Scrape webpages url using Firecrawl and return the contents"
    args_schema: Type[BaseModel] = FirecrawlScrapeWebsiteToolSchema
    api_key: Optional[str] = None
    firecrawl: Optional[Any] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from firecrawl import FirecrawlApp # type: ignore
        except ImportError:
           raise ImportError(
               "`firecrawl` package not found, please run `pip install firecrawl-py`"
           )

        self.firecrawl = FirecrawlApp(api_key=api_key)

    def _run(self, url: str, page_options: Optional[Dict[str, Any]] = None, extractor_options: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None):
        options = {
            "pageOptions": page_options,
            "extractorOptions": extractor_options,
            "timeout": timeout
        }
        return self.firecrawl.scrape_url(url, options)