from typing import Any, Optional, Type

from pydantic import BaseModel, Field
from .serpapi_base_tool import SerpApiBaseTool
from urllib.error import HTTPError

from .serpapi_base_tool import SerpApiBaseTool


class SerpApiGoogleShoppingToolSchema(BaseModel):
    """Input for Google Shopping."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to Google shopping."
    )
    location: Optional[str] = Field(
        None, description="Location you want the search to be performed in."
    )


class SerpApiGoogleShoppingTool(SerpApiBaseTool):
    name: str = "Google Shopping"
    description: str = (
        "A tool to perform search on Google shopping with a search_query."
    )
    args_schema: Type[BaseModel] = SerpApiGoogleShoppingToolSchema

    def _run(
        self,
        **kwargs: Any,
    ) -> Any:
        try:
            results = self.client.search(
                {
                    "engine": "google_shopping",
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
                    r"filters",
                    r"pagination",
                ],
            )

            return results
        except HTTPError as e:
            return f"An error occurred: {str(e)}. Some parameters may be invalid."
