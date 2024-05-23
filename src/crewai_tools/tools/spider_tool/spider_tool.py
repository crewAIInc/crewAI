from typing import Optional, Any, Type, Dict, Literal
from pydantic.v1 import BaseModel, Field
from crewai_tools.tools.base_tool import BaseTool

class SpiderToolSchema(BaseModel):
    url: str = Field(description="Website URL")
    params: Optional[Dict[str, Any]] = Field(default={"return_format": "markdown"}, description="Set additional params. Leave empty for this to return LLM-ready data")
    mode: Optional[Literal["scrape", "crawl"]] = Field(defualt="scrape", description="Mode, the only two allowed modes are `scrape` or `crawl` the url")

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
        params: Optional[Dict[str, any]] = None,
        mode: Optional[Literal["scrape", "crawl"]] = "scrape"
    ):
        if mode not in ["scrape", "crawl"]:
            raise ValueError(
                "Unknown mode in `mode` parameter, `scrape` or `crawl` are the allowed modes"
            )

        if params is None or params == {}:
            params = {"return_format": "markdown"}

        action = (
            self.spider.scrape_url if mode == "scrape" else self.spider.crawl_url
        )
        spider_docs = action(url=url, params=params)

        return spider_docs
