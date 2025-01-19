import os
from typing import Any, Optional, Type, Dict, Literal, Union

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class HyperbrowserLoadToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    operation: Literal['scrape', 'crawl'] = Field(description="Operation to perform on the website. Either 'scrape' or 'crawl'")
    params: Optional[Dict] = Field(description="Optional params for scrape or crawl. For more information on the supported params, visit https://docs.hyperbrowser.ai/reference/sdks/python/scrape#start-scrape-job-and-wait or https://docs.hyperbrowser.ai/reference/sdks/python/crawl#start-crawl-job-and-wait")

class HyperbrowserLoadTool(BaseTool):
    """HyperbrowserLoadTool.

    Scrape or crawl web pages and load the contents with optional parameters for configuring content extraction.
    Requires the `hyperbrowser` package.
    Get your API Key from https://app.hyperbrowser.ai/

    Args:
        api_key: The Hyperbrowser API key, can be set as an environment variable `HYPERBROWSER_API_KEY` or passed directly
    """
    name: str = "Hyperbrowser web load tool"
    description: str = "Scrape or crawl a website using Hyperbrowser and return the contents in properly formatted markdown or html"
    args_schema: Type[BaseModel] = HyperbrowserLoadToolSchema
    api_key: Optional[str] = None
    hyperbrowser: Optional[Any] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv('HYPERBROWSER_API_KEY')
        if not api_key:
            raise ValueError(
                "`api_key` is required, please set the `HYPERBROWSER_API_KEY` environment variable or pass it directly"
            )

        try:
            from hyperbrowser import Hyperbrowser
        except ImportError:
            raise ImportError("`hyperbrowser` package not found, please run `pip install hyperbrowser`")

        if not self.api_key:
            raise ValueError("HYPERBROWSER_API_KEY is not set. Please provide it either via the constructor with the `api_key` argument or by setting the HYPERBROWSER_API_KEY environment variable.")

        self.hyperbrowser = Hyperbrowser(api_key=self.api_key)

    def _prepare_params(self, params: Dict) -> Dict:
        """Prepare session and scrape options parameters."""
        try:
            from hyperbrowser.models.session import CreateSessionParams
            from hyperbrowser.models.scrape import ScrapeOptions
        except ImportError:
            raise ImportError(
                "`hyperbrowser` package not found, please run `pip install hyperbrowser`"
            )

        if "scrape_options" in params:
            if "formats" in params["scrape_options"]:
                formats = params["scrape_options"]["formats"]
                if not all(fmt in ["markdown", "html"] for fmt in formats):
                    raise ValueError("formats can only contain 'markdown' or 'html'")

        if "session_options" in params:
            params["session_options"] = CreateSessionParams(**params["session_options"])
        if "scrape_options" in params:
            params["scrape_options"] = ScrapeOptions(**params["scrape_options"])
        return params
    
    def _extract_content(self, data: Union[Any, None]):
        """Extract content from response data."""
        content = ""
        if data:
            content = data.markdown or data.html or ""
        return content

    def _run(self, url: str, operation: Literal['scrape', 'crawl'] = 'scrape', params: Optional[Dict] = {}):
        try:
            from hyperbrowser.models.scrape import StartScrapeJobParams
            from hyperbrowser.models.crawl import StartCrawlJobParams
        except ImportError:
            raise ImportError(
                "`hyperbrowser` package not found, please run `pip install hyperbrowser`"
            )
        
        params = self._prepare_params(params)

        if operation == 'scrape':
            scrape_params = StartScrapeJobParams(url=url, **params)
            scrape_resp = self.hyperbrowser.scrape.start_and_wait(scrape_params)
            content = self._extract_content(scrape_resp.data)
            return content
        else:
            crawl_params = StartCrawlJobParams(url=url, **params)
            crawl_resp = self.hyperbrowser.crawl.start_and_wait(crawl_params)
            content = ""
            if crawl_resp.data:
                for page in crawl_resp.data:
                    page_content = self._extract_content(page)
                    if page_content:
                        content += (
                            f"\n{'-'*50}\nUrl: {page.url}\nContent:\n{page_content}\n"
                        )
            return content
