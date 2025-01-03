import logging
from typing import Any, Dict, Literal, Optional, Type
from urllib.parse import unquote, urlparse

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


class SpiderToolConfig(BaseModel):
    """Configuration settings for SpiderTool.

    Contains all default values and constants used by SpiderTool.
    Centralizes configuration management for easier maintenance.
    """

    # Crawling settings
    DEFAULT_CRAWL_LIMIT: int = 5
    DEFAULT_RETURN_FORMAT: str = "markdown"

    # Request parameters
    DEFAULT_REQUEST_MODE: str = "smart"
    FILTER_SVG: bool = True


class SpiderTool(BaseTool):
    """Tool for scraping and crawling websites.
    This tool provides functionality to either scrape a single webpage or crawl multiple
    pages, returning content in a format suitable for LLM processing.
    """

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
    config: SpiderToolConfig = SpiderToolConfig()

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
                "`spider-client` package not found, please run `uv add spider-client`"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Spider client: {str(e)}")

    def _validate_url(self, url: str) -> bool:
        """Validate URL format and security constraints.

        Args:
            url (str): URL to validate. Must be a properly formatted HTTP(S) URL

        Returns:
            bool: True if URL is valid and meets security requirements, False otherwise.
        """
        try:
            url = url.strip()
            decoded_url = unquote(url)

            result = urlparse(decoded_url)
            if not all([result.scheme, result.netloc]):
                return False

            if result.scheme not in ["http", "https"]:
                return False

            return True
        except Exception:
            return False

    def _run(
        self,
        website_url: str,
        mode: Literal["scrape", "crawl"] = "scrape",
    ) -> Optional[str]:
        """Execute the spider tool to scrape or crawl the specified website.

        Args:
            website_url (str): The URL to process. Must be a valid HTTP(S) URL.
            mode (Literal["scrape", "crawl"]): Operation mode.
                - "scrape": Extract content from single page
                - "crawl": Follow links and extract content from multiple pages

        Returns:
            Optional[str]: Extracted content in markdown format, or None if extraction fails
                        and log_failures is True.

        Raises:
            ValueError: If URL is invalid or missing, or if mode is invalid.
            ImportError: If spider-client package is not properly installed.
            ConnectionError: If network connection fails while accessing the URL.
            Exception: For other runtime errors.
        """

        try:
            params = {}
            url = website_url or self.website_url

            if not url:
                raise ValueError(
                    "Website URL must be provided either during initialization or execution"
                )

            if not self._validate_url(url):
                raise ValueError(f"Invalid URL format: {url}")

            if mode not in ["scrape", "crawl"]:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be either 'scrape' or 'crawl'"
                )

            params = {
                "request": self.config.DEFAULT_REQUEST_MODE,
                "filter_output_svg": self.config.FILTER_SVG,
                "return_format": self.config.DEFAULT_RETURN_FORMAT,
            }

            if mode == "crawl":
                params["limit"] = self.config.DEFAULT_CRAWL_LIMIT

            if self.custom_params:
                params.update(self.custom_params)

            action = (
                self.spider.scrape_url if mode == "scrape" else self.spider.crawl_url
            )
            return action(url=url, params=params)

        except ValueError as ve:
            if self.log_failures:
                logger.error(f"Validation error for URL {url}: {str(ve)}")
                return None
            raise ve

        except ImportError as ie:
            logger.error(f"Spider client import error: {str(ie)}")
            raise ie

        except ConnectionError as ce:
            if self.log_failures:
                logger.error(f"Connection error while accessing {url}: {str(ce)}")
                return None
            raise ce

        except Exception as e:
            if self.log_failures:
                logger.error(
                    f"Unexpected error during {mode} operation on {url}: {str(e)}"
                )
                return None
            raise e
