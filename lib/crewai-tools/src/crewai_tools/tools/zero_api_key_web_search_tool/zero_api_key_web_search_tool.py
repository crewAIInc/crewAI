"""Zero-API-Key Web Search tool for CrewAI.

Free by default (DuckDuckGo), with optional production-grade
Bright Data SERP supporting 7 search engines (Google, Bing,
DuckDuckGo, Yandex, Baidu, Yahoo, Naver) and geo-targeting
for 195 countries.

Install the package first:
    pip install zero-api-key-web-search

Example:
    ```python
    from crewai_tools import ZeroApiKeyWebSearchTool

    # Free search — no API key needed
    tool = ZeroApiKeyWebSearchTool()
    result = tool.run(search_query="Python 3.13 release")

    # With Bright Data for production SERP
    tool = ZeroApiKeyWebSearchTool(profile="production")
    result = tool.run(search_query="AI regulation news")
    ```
"""

from typing import Any, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ZeroApiKeyWebSearchToolSchema(BaseModel):
    """Input schema for Zero-API-Key Web Search tool."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use to search the internet",
    )


class ZeroApiKeyWebSearchTool(BaseTool):
    """Search the web using Zero-API-Key Web Search.

    Free by default (DuckDuckGo), with optional production-grade
    Bright Data SERP supporting 7 search engines and geo-targeting
    for 195 countries.

    Also supports claim verification and page reading via the
    underlying package.
    """

    name: str = "Search the web with Zero-API-Key Web Search"
    description: str = (
        "Search the web for real-time information using Zero-API-Key Web Search. "
        "Free by default (DuckDuckGo). Optionally use Bright Data for production-grade "
        "multi-engine SERP (Google, Bing, DuckDuckGo, Yandex, Baidu, Yahoo, Naver) "
        "with geo-targeting for 195 countries. Returns search results with titles, "
        "URLs, snippets, and an answer summary."
    )
    args_schema: type[BaseModel] = ZeroApiKeyWebSearchToolSchema
    profile: str = "default"
    search_type: str = "text"
    region: str = "wt-wt"
    max_results: int = 10
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["zero-api-key-web-search"],
    )

    def _run(self, **kwargs: Any) -> str:
        search_query = kwargs.get("search_query") or kwargs.get("query", "")
        if not search_query:
            return "Error: search_query is required."

        try:
            from zero_api_key_web_search import UltimateSearcher
        except ImportError:
            return (
                "Error: zero-api-key-web-search is not installed. "
                "Install it with: pip install zero-api-key-web-search"
            )

        searcher = UltimateSearcher(profile=self.profile)
        answer = searcher.search(
            query=search_query,
            search_type=self.search_type,
            region=self.region,
            max_results=self.max_results,
        )

        if not answer.sources:
            return f"No results found for: {search_query}"

        result = f"## Search Results for: {search_query}\n\n"
        result += f"**Answer:** {answer.answer}\n\n"
        result += "### Sources:\n\n"
        for i, source in enumerate(answer.sources[:self.max_results], 1):
            result += f"{i}. **{source.title}**\n"
            result += f"   [{source.url}]({source.url})\n"
            if source.snippet:
                result += f"   {source.snippet}\n"
            if source.date:
                result += f"   Date: {source.date}\n"
            result += "\n"

        providers_used = ", ".join(answer.metadata.get("providers_used", []))
        if providers_used:
            result += f"Providers: {providers_used}\n"

        return result