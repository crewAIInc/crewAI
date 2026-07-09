from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class DuckDuckGoSearchToolSchema(BaseModel):
    """Input for DuckDuckGoSearchTool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the internet",
    )


class DuckDuckGoSearchTool(BaseTool):
    """
    DuckDuckGoSearchTool - Search the internet using DuckDuckGo.

    Unlike most search tools in this repository, DuckDuckGo requires **no API key**,
    which makes it a convenient, zero-configuration default for web search. It is
    powered by the `ddgs` package.

    Dependencies:
        - ddgs
    """

    name: str = "DuckDuckGo Web Search"
    description: str = (
        "A tool that searches the internet using DuckDuckGo. "
        "It requires no API key. Provide a 'search_query' to retrieve relevant "
        "web results (title, link and snippet)."
    )
    args_schema: type[BaseModel] = DuckDuckGoSearchToolSchema
    n_results: int = 10
    region: str = "wt-wt"
    safesearch: str = "moderate"
    package_dependencies: list[str] = Field(default_factory=lambda: ["ddgs"])

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            from ddgs import DDGS  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Missing optional dependency 'ddgs'. Install it with:\n"
                "  uv add crewai-tools --extra ddgs\n"
                "or\n"
                "  pip install ddgs\n"
            ) from exc

    def _run(self, **kwargs: Any) -> str:
        search_query = kwargs.get("search_query") or kwargs.get("query")
        if not search_query:
            return "Error: a 'search_query' is required to perform a DuckDuckGo search."

        n_results = kwargs.get("n_results", self.n_results)
        region = kwargs.get("region", self.region)
        safesearch = kwargs.get("safesearch", self.safesearch)

        from ddgs import DDGS

        try:
            results = DDGS().text(
                search_query,
                region=region,
                safesearch=safesearch,
                max_results=n_results,
            )
        except Exception as e:
            return f"Error performing DuckDuckGo search: {e}"

        if not results:
            return f"No results found for '{search_query}'."

        formatted = [
            "\n".join(
                [
                    f"Title: {result.get('title', '')}",
                    f"Link: {result.get('href', '')}",
                    f"Snippet: {result.get('body', '')}",
                    "---",
                ]
            )
            for result in results
        ]
        return "\n".join(formatted)
