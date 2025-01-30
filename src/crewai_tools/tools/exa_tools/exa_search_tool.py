from typing import Any, Optional, Type
from pydantic import BaseModel, Field

try:
    from exa_py import Exa

    EXA_INSTALLED = True
except ImportError:
    Exa = Any
    EXA_INSTALLED = False


class EXABaseToolToolSchema(BaseModel):
    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )


class EXASearchTool:
    args_schema: Type[BaseModel] = EXABaseToolToolSchema
    client: Optional["Exa"] = Field(default=None, description="Exa search client")

    def __init__(
        self,
        api_key: str,
        content: bool = False,
        highlights: bool = False,
        type: str = "keyword",
        use_autoprompt: bool = True,
    ):
        if not EXA_INSTALLED:
            raise ImportError("`exa-py` package not found, please run `uv add exa-py`")
        self.client = Exa(api_key=api_key)
        self.content = content
        self.highlights = highlights
        self.type = type
        self.use_autoprompt = use_autoprompt

    def _run(
        self,
        search_query: str,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        include_domains: Optional[list[str]] = None,
    ) -> Any:
        if self.client is None:
            raise ValueError("Client not initialized")

        search_params = {
            "use_autoprompt": self.use_autoprompt,
            "type": self.type,
        }

        if start_published_date:
            search_params["start_published_date"] = start_published_date
        if end_published_date:
            search_params["end_published_date"] = end_published_date
        if include_domains:
            search_params["include_domains"] = include_domains

        if self.content:
            results = self.client.search_and_contents(
                search_query, highlights=self.highlights, **search_params
            )
        else:
            results = self.client.search(search_query, **search_params)
        return results
