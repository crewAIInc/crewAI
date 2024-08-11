from typing import Optional, Any, Type, Dict, Literal
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

class SpiderToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    params: Optional[Dict[str, Any]] = Field(
        description="Set additional params. Options include:\n"
                    "- `limit`: Optional[int] - The maximum number of pages allowed to crawl per website. Remove the value or set it to `0` to crawl all pages.\n"
                    "- `depth`: Optional[int] - The crawl limit for maximum depth. If `0`, no limit will be applied.\n"
                    "- `metadata`: Optional[bool] - Boolean to include metadata or not. Defaults to `False` unless set to `True`. If the user wants metadata, include params.metadata = True.\n"
                    "- `query_selector`: Optional[str] - The CSS query selector to use when extracting content from the markup.\n"
    )
    mode: Literal["scrape", "crawl"] = Field(
        default="scrape",
        description="Mode, the only two allowed modes are `scrape` or `crawl`. Use `scrape` to scrape a single page and `crawl` to crawl the entire website following subpages. These modes are the only allowed values even when ANY params is set."
    )

class SpiderTool(BaseTool):
    name: str = "Spider scrape & crawl tool"
    description: str = "Scrape & Crawl any url and return LLM-ready data."
    args_schema: Type[BaseModel] = SpiderToolSchema
    api_key: Optional[str] = None
    spider: Optional[Any] = None

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        try:
            from spider import Spider # type: ignore
        except ImportError:
           raise ImportError(
               "`spider-client` package not found, please run `pip install spider-client`"
           )

        self.spider = Spider(api_key=api_key)

    def _run(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        mode: Optional[Literal["scrape", "crawl"]] = "scrape"
    ):
        if mode not in ["scrape", "crawl"]:
            raise ValueError(
                "Unknown mode in `mode` parameter, `scrape` or `crawl` are the allowed modes"
            )

        # Ensure 'return_format': 'markdown' is always included
        if params:
            params["return_format"] = "markdown"
        else:
            params = {"return_format": "markdown"}

        action = (
            self.spider.scrape_url if mode == "scrape" else self.spider.crawl_url
        )
        spider_docs = action(url=url, params=params)

        return spider_docs
