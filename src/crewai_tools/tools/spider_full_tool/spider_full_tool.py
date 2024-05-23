from typing import Optional, Any, Type, Dict, Literal
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool
import requests

class SpiderFullParams(BaseModel):
    request: Optional[str] = Field(description="The request type to perform. Possible values are `http`, `chrome`, and `smart`.")
    limit: Optional[int] = Field(description="The maximum number of pages allowed to crawl per website. Remove the value or set it to `0` to crawl all pages.")
    depth: Optional[int] = Field(description="The crawl limit for maximum depth. If `0`, no limit will be applied.")
    cache: Optional[bool] = Field(default=True, description="Use HTTP caching for the crawl to speed up repeated runs.")
    budget: Optional[Dict[str, int]] = Field(description="Object that has paths with a counter for limiting the number of pages, e.g., `{'*':1}` for only crawling the root page.")
    locale: Optional[str] = Field(description="The locale to use for request, e.g., `en-US`.")
    cookies: Optional[str] = Field(description="Add HTTP cookies to use for request.")
    stealth: Optional[bool] = Field(default=True, description="Use stealth mode for headless chrome request to help prevent being blocked. Default is `true` on chrome.")
    headers: Optional[Dict[str, str]] = Field(description="Forward HTTP headers to use for all requests. The object is expected to be a map of key-value pairs.")
    metadata: Optional[bool] = Field(default=False, description="Boolean to store metadata about the pages and content found. Defaults to `false` unless enabled.")
    viewport: Optional[str] = Field(default="800x600", description="Configure the viewport for chrome. Defaults to `800x600`.")
    encoding: Optional[str] = Field(description="The type of encoding to use, e.g., `UTF-8`, `SHIFT_JIS`.")
    subdomains: Optional[bool] = Field(default=False, description="Allow subdomains to be included. Default is `false`.")
    user_agent: Optional[str] = Field(description="Add a custom HTTP user agent to the request. Default is a random agent.")
    store_data: Optional[bool] = Field(default=False, description="Boolean to determine if storage should be used. Defaults to `false`.")
    gpt_config: Optional[Dict[str, Any]] = Field(description="Use AI to generate actions to perform during the crawl. Can pass an array for the `prompt` to chain steps.")
    fingerprint: Optional[bool] = Field(description="Use advanced fingerprinting for chrome.")
    storageless: Optional[bool] = Field(default=False, description="Boolean to prevent storing any data for the request. Defaults to `false`.")
    readability: Optional[bool] = Field(description="Use readability to pre-process the content for reading.")
    return_format: Optional[str] = Field(default="markdown", description="The format to return the data in. Possible values are `markdown`, `raw`, `text`, and `html2text`.")
    proxy_enabled: Optional[bool] = Field(description="Enable high-performance premium proxies to prevent being blocked.")
    query_selector: Optional[str] = Field(description="The CSS query selector to use when extracting content from the markup.")
    full_resources: Optional[bool] = Field(description="Crawl and download all resources for a website.")
    request_timeout: Optional[int] = Field(default=30, description="The timeout for requests. Ranges from `5-60` seconds. Default is `30` seconds.")
    run_in_background: Optional[bool] = Field(description="Run the request in the background. Useful if storing data and triggering crawls to the dashboard.")

class SpiderFullToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    params: Optional[SpiderFullParams] = Field(default=SpiderFullParams(), description="All the params available")
    mode: Optional[Literal["scrape", "crawl"]] = Field(default="scrape", description="Mode, either `scrape` or `crawl` the URL")

class SpiderFullTool(BaseTool):
    name: str = "Spider scrape & crawl tool"
    description: str = "Scrape & Crawl any URL and return LLM-ready data."
    args_schema: Type[BaseModel] = SpiderFullToolSchema
    api_key: Optional[str] = None
    spider: Optional[Any] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from spider import Spider  # type: ignore
        except ImportError:
            raise ImportError(
                "`spider-client` package not found, please run `pip install spider-client`"
            )

        self.spider = Spider(api_key=api_key)

    def _run(
        self,
        url: str,
        params: Optional[SpiderFullParams] = None,
        mode: Optional[Literal["scrape", "crawl"]] = "scrape"
    ):
        if mode not in ["scrape", "crawl"]:
            raise ValueError(
                "Unknown mode in `mode` parameter, `scrape` or `crawl` are the allowed modes"
            )

        if params is None:
            print("PARAMS IT NONE")
            params = SpiderFullParams()
            print(params)

        action = self.spider.scrape_url if mode == "scrape" else self.spider.crawl_url
        response = action(url=url, params=params.dict())

        # Debugging: Print the response content
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.text}")

        try:
            spider_docs = response.json()
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            spider_docs = {"error": "Failed to decode JSON response"}

        return spider_docs
