from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import requests

from crewai_tools.tools.searchapi_tool.searchapi_base_tool import SearchApiBaseTool


OMIT_FIELDS = ["search_metadata", "search_parameters", "pagination"]


class SearchApiGoogleShoppingToolSchema(BaseModel):
    """Input for Google Shopping."""

    search_query: str = Field(
        ...,
        description="Mandatory search query you want to use for Google Shopping.",
    )
    location: str | None = Field(
        None, description="Location you want the search to be performed in."
    )


class SearchApiGoogleShoppingTool(SearchApiBaseTool):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "SearchApi Google Shopping"
    description: str = "A tool to perform a Google Shopping search with a search_query using SearchApi."
    args_schema: type[BaseModel] = SearchApiGoogleShoppingToolSchema

    def _run(self, **kwargs: Any) -> Any:
        try:
            params = {
                "engine": "google_shopping",
                "q": kwargs.get("search_query"),
            }
            location = kwargs.get("location")
            if location:
                params["location"] = location

            results = self._search(params)
            return self._omit_fields(results, OMIT_FIELDS)
        except requests.RequestException as e:
            return f"An error occurred while performing the search: {e!s}"
