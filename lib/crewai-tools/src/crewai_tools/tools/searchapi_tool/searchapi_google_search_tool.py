from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import requests

from crewai_tools.tools.searchapi_tool.searchapi_base_tool import SearchApiBaseTool


OMIT_FIELDS = ["search_metadata", "search_parameters", "pagination"]


class SearchApiGoogleSearchToolSchema(BaseModel):
    """Input for Google Search."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to Google search."
    )
    location: str | None = Field(
        None, description="Location you want the search to be performed in."
    )


class SearchApiGoogleSearchTool(SearchApiBaseTool):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "SearchApi Google Search"
    description: str = (
        "A tool to perform a Google search with a search_query using SearchApi."
    )
    args_schema: type[BaseModel] = SearchApiGoogleSearchToolSchema

    def _run(self, **kwargs: Any) -> Any:
        try:
            params = {
                "engine": "google",
                "q": kwargs.get("search_query"),
            }
            location = kwargs.get("location")
            if location:
                params["location"] = location

            results = self._search(params)
            return self._omit_fields(results, OMIT_FIELDS)
        except requests.HTTPError as e:
            return f"An error occurred: {e!s}. Some parameters may be invalid."
