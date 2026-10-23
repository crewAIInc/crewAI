from typing import Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.geekflare_tool.geekflare_base import GeekflareBaseTool


class GeekflareSearchToolSchema(BaseModel):
    """Input schema for GeekflareSearchTool."""

    query: str = Field(..., description="The search query.")


class GeekflareSearchTool(GeekflareBaseTool):
    """Search the web, news, or images with Geekflare, or fetch the full Google SERP.

    Attributes:
        limit: Number of results (1-100).
        source: web, news, or images.
        location: Country code (ISO alpha-2) for localized results.
        city: City for localized results, e.g. "London,England,United Kingdom".
        time: Recency filter: any, h, d, w, m, y, or with a count such as d7.
        include_domains: Only return results from these domains.
        exclude_domains: Exclude results from these domains.
        serp: Return the full Google results page (organic results, AI Overview,
            People Also Ask, related searches).
        grounded_answer: Add an AI answer grounded in the results (ignored with serp).
    """

    name: str = "Geekflare Search"
    description: str = (
        "Search the live web with Geekflare and return results with titles, URLs, and "
        "snippets as JSON. Use for current information and for finding sources."
    )
    args_schema: type[BaseModel] = GeekflareSearchToolSchema
    limit: int = Field(default=10, ge=1, le=100, description="Number of results.")
    source: Literal["web", "news", "images"] = Field(
        default="web", description="Search source."
    )
    location: str | None = Field(default=None, description="Country code, e.g. us.")
    city: str | None = Field(default=None, description="City for localized results.")
    time: str | None = Field(default=None, description="Recency filter.")
    include_domains: list[str] | None = Field(
        default=None, description="Only return results from these domains."
    )
    exclude_domains: list[str] | None = Field(
        default=None, description="Exclude results from these domains."
    )
    serp: bool = Field(default=False, description="Return the full Google SERP.")
    grounded_answer: bool = Field(
        default=False, description="Add an AI answer grounded in the results."
    )

    def _run(self, query: str) -> str:
        return self._post_json(
            "search",
            {
                "query": query,
                "limit": self.limit,
                "source": self.source,
                "location": self.location,
                "city": self.city,
                "time": self.time,
                "includeDomains": self.include_domains,
                "excludeDomains": self.exclude_domains,
                "serp": self.serp,
                "groundedAnswer": self.grounded_answer,
            },
        )
