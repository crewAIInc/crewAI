from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from crewai_tools.tools.base_tool import BaseTool


class FirecrawlSearchToolSchema(BaseModel):
    query: str = Field(description="Search query")
    page_options: Optional[Dict[str, Any]] = Field(
        default=None, description="Options for result formatting"
    )
    search_options: Optional[Dict[str, Any]] = Field(
        default=None, description="Options for searching"
    )


class FirecrawlSearchTool(BaseTool):
    name: str = "Firecrawl web search tool"
    description: str = "Search webpages using Firecrawl and return the results"
    args_schema: Type[BaseModel] = FirecrawlSearchToolSchema
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
        query: str,
        page_options: Optional[Dict[str, Any]] = None,
        result_options: Optional[Dict[str, Any]] = None,
    ):
        if page_options is None:
            page_options = {}
        if result_options is None:
            result_options = {}

        options = {"pageOptions": page_options, "resultOptions": result_options}
        return self.firecrawl.search(query, options)
