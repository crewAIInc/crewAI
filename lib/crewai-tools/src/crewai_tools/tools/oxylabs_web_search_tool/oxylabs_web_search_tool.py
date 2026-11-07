import json

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from crewai_tools.tools.oxylabs_web_api_base_tool.oxylabs_web_api_base_tool import (
    OxylabsWebAPIBaseTool,
)


__all__ = ["OxylabsWebSearchTool"]


class OxylabsWebSearchArgs(BaseModel):
    query: str = Field(
        description=(
            "What to search for, as a natural search phrase. Keyword-shaped queries "
            "beat sentence-shaped ones. One question per search."
        ),
        min_length=1,
        max_length=2048,
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return. The response may contain fewer.",
        ge=1,
        le=20,
    )
    location: str | None = Field(
        default=None,
        description=(
            "ISO 3166-1 alpha-2 country code to rank results for, e.g. DE, US, LT. "
            "Pass it whenever the answer is geographic. A place name such as "
            "'Germany' is rejected."
        ),
        pattern="^[A-Za-z]{2}$",
    )


class OxylabsWebSearchTool(OxylabsWebAPIBaseTool):
    """Search the live web with the Oxylabs Web API.

    Get an API key from a Web API instance in the Oxylabs dashboard:
    https://dashboard.oxylabs.io

    Args:
        api_key (str): Oxylabs Web API key. Defaults to ``OXYLABS_WEB_API_KEY``.
    """

    name: str = "Oxylabs Web Search tool"
    description: str = (
        "Search the live web and return ranked organic results. Use it for anything "
        "where being out of date makes the answer wrong: current events, news, "
        "prices, availability, versions, rankings. Results are ranked as someone in "
        "the `location` country would see them. Returns titles, URLs and short "
        "snippets, not page content: read the pages worth reading with the Oxylabs "
        "Web Scrape tool."
    )
    args_schema: type[BaseModel] = OxylabsWebSearchArgs

    def _run(
        self, query: str, max_results: int = 10, location: str | None = None
    ) -> str | ToolFailure:
        if not query.strip():
            return self._invalid_input("`query` must not be empty.")

        payload: dict[str, object] = {"query": query, "max_results": max_results}
        if location:
            payload["location"] = location.strip().upper()

        response = self._request("POST", "/v1/search", payload)
        if isinstance(response, ToolFailure):
            return response

        results = [
            {key: item[key] for key in ("title", "url", "outline") if key in item}
            for item in response.get("results") or []
            if isinstance(item, dict)
        ]
        questions = [
            item["question"]
            for item in response.get("related_questions") or []
            if isinstance(item, dict) and item.get("question")
        ]
        out: dict[str, object] = {"results": results}
        if questions:
            out["related_questions"] = questions
        return json.dumps(out, ensure_ascii=False)
