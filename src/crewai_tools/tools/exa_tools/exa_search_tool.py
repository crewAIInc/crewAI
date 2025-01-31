from typing import Any, Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

try:
    from exa_py import Exa

    EXA_INSTALLED = True
except ImportError:
    Exa = Any
    EXA_INSTALLED = False


class EXABaseToolSchema(BaseModel):
    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )
    start_published_date: Optional[str] = Field(
        None, description="Start date for the search"
    )
    end_published_date: Optional[str] = Field(
        None, description="End date for the search"
    )
    include_domains: Optional[list[str]] = Field(
        None, description="List of domains to include in the search"
    )


class EXASearchTool(BaseTool):
    model_config = {"arbitrary_types_allowed": True}
    name: str = "EXASearchTool"
    description: str = "Search the internet using Exa"
    args_schema: Type[BaseModel] = EXABaseToolSchema
    client: Optional["Exa"] = None
    content: Optional[bool] = False
    summary: Optional[bool] = False
    type: Optional[str] = "auto"

    def __init__(
        self,
        api_key: str,
        content: Optional[bool] = False,
        summary: Optional[bool] = False,
        type: Optional[str] = "auto",
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        if not EXA_INSTALLED:
            import click

            if click.confirm(
                "You are missing the 'exa_py' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "exa_py"], check=True)

            else:
                raise ImportError(
                    "You are missing the 'exa_py' package. Would you like to install it?"
                )
        self.client = Exa(api_key=api_key)
        self.content = content
        self.summary = summary
        self.type = type

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
                search_query, summary=self.summary, **search_params
            )
        else:
            results = self.client.search(search_query, **search_params)
        return results
