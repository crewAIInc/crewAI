import logging
from typing import Any, Dict, Literal, Optional, Type
from urllib.parse import urlparse

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__file__)


class SpiderToolSchema(BaseModel):
    """Input schema for SpiderTool."""

    website_url: str = Field(
        ..., description="Mandatory website URL to scrape or crawl"
    )
    mode: Literal["scrape", "crawl"] = Field(
        default="scrape",
        description="The mode of the SpiderTool. The only two allowed modes are `scrape` or `crawl`. Crawl mode will follow up to 5 links and return their content in markdown format.",
    )


class SpiderTool(BaseTool):
    """Tool for scraping and crawling websites."""

    DEFAULT_CRAWL_LIMIT: int = 5
    DEFAULT_RETURN_FORMAT: str = "markdown"

    name: str = "SpiderTool"
    description: str = (
        "A tool to scrape or crawl a website and return LLM-ready content."
    )
    args_schema: Type[BaseModel] = SpiderToolSchema
    custom_params: Optional[Dict[str, Any]] = None
    website_url: Optional[str] = None
    api_key: Optional[str] = None
    spider: Any = None
    log_failures: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        website_url: Optional[str] = None,
        custom_params: Optional[Dict[str, Any]] = None,
        log_failures: bool = True,
        **kwargs,
    ):
        """Initialize SpiderTool for web scraping and crawling.

        Args:
            api_key (Optional[str]): Spider API key for authentication. Required for production use.
            website_url (Optional[str]): Default website URL to scrape/crawl. Can be overridden during execution.
            custom_params (Optional[Dict[str, Any]]): Additional parameters to pass to Spider API.
                These override any parameters set by the LLM.
            log_failures (bool): If True, logs errors. Defaults to True.
            **kwargs: Additional arguments passed to BaseTool.

        Raises:
            ImportError: If spider-client package is not installed.
            RuntimeError: If Spider client initialization fails.
        """

        super().__init__(**kwargs)
        if website_url is not None:
            self.website_url = website_url

        self.log_failures = log_failures
        self.custom_params = custom_params

        try:
            from spider import Spider  # type: ignore

            self.spider = Spider(api_key=api_key)
        except ImportError:
            raise ImportError(
                "`spider-client` package not found, please run `pip install spider-client`"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Spider client: {str(e)}")

    def _validate_url(self, url: str) -> bool:
        """Validate URL format.

        Args:
            url (str): URL to validate.
        Returns:
            bool: True if valid URL.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _run(
        self,
        website_url: str,
        mode: Literal["scrape", "crawl"] = "scrape",
    ) -> str:
        params = {}
        url = website_url or self.website_url

        if not url:
            raise ValueError(
                "Website URL must be provided either during initialization or execution"
            )

        if not self._validate_url(url):
            raise ValueError("Invalid URL format")

        if mode not in ["scrape", "crawl"]:
            raise ValueError("Mode must be either 'scrape' or 'crawl'")

        params["request"] = "smart"
        params["filter_output_svg"] = True
        params["return_format"] = self.DEFAULT_RETURN_FORMAT

        if mode == "crawl":
            params["limit"] = self.DEFAULT_CRAWL_LIMIT

        # Update params with custom params if provided.
        # This will override any params passed by LLM.
        if self.custom_params:
            params.update(self.custom_params)

        try:
            action = (
                self.spider.scrape_url if mode == "scrape" else self.spider.crawl_url
            )
            return action(url=url, params=params)

        except Exception as e:
            if self.log_failures:
                logger.error(f"Error fetching data from {url}, exception: {e}")
                return None
            else:
                raise e
