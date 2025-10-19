from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crewai_tools.tools.serpapi_tool.serpapi_base_tool import SerpApiBaseTool


try:
    from serpapi import HTTPError  # type: ignore[import-untyped]
except ImportError:
    HTTPError = Any


class SerpApiGoogleSearchToolSchema(BaseModel):
    """Input for Google Search."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to Google search."
    )
    location: str | None = Field(
        None, description="Location you want the search to be performed in."
    )


class SerpApiGoogleSearchTool(SerpApiBaseTool):
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Google Search"
    description: str = (
        "A tool to perform to perform a Google search with a search_query."
    )
    args_schema: type[BaseModel] = SerpApiGoogleSearchToolSchema

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        try:
            results = self.client.search(  # type: ignore[union-attr]
                {
                    "q": kwargs.get("search_query"),
                    "location": kwargs.get("location"),
                }
            ).as_dict()

            self._omit_fields(
                results,
                [
                    r"search_metadata",
                    r"search_parameters",
                    r"serpapi_.+",
                    r".+_token",
                    r"displayed_link",
                    r"pagination",
                ],
            )

            return results
        except HTTPError as e:
            return f"An error occurred: {e!s}. Some parameters may be invalid."
