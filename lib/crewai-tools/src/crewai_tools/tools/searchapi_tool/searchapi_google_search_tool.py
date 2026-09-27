import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
import requests

from crewai_tools.tools.searchapi_tool.searchapi_base_tool import SearchApiBaseTool


class SearchApiGoogleSearchToolSchema(BaseModel):
    """Input for SearchApi Google Search."""

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

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        self._validate_url()
        api_key = os.getenv("SEARCHAPI_API_KEY")
        query = kwargs.get("search_query") or kwargs.get("q")
        params: dict[str, Any] = {
            "engine": "google",
            "q": query,
        }
        if kwargs.get("location"):
            params["location"] = kwargs.get("location")

        try:
            response = requests.get(
                self.url,
                headers={"Authorization": f"Bearer {api_key}"},
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()

            self._omit_fields(
                results,
                [
                    r"search_metadata",
                    r"search_parameters",
                ],
            )

            return results
        except requests.RequestException as e:
            return f"An error occurred: {e!s}. Some parameters may be invalid."
