import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field
import requests


class ShopSavvyProductSearchToolSchema(BaseModel):
    """Input schema for ShopSavvyProductSearchTool."""

    query: str = Field(
        ...,
        description="Search query: product name, barcode, UPC, EAN, ISBN, ASIN, or URL.",
    )


class ShopSavvyProductSearchTool(BaseTool):
    """Tool that searches for products and retrieves real-time pricing from
    thousands of retailers using the ShopSavvy Data API.

    Attributes:
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The ShopSavvy API key.
        base_url: The base URL for the ShopSavvy API.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "ShopSavvy Product Search"
    description: str = (
        "Search for products and get real-time pricing across thousands of retailers "
        "using the ShopSavvy API. Accepts product names, barcodes, UPCs, ASINs, or URLs as queries."
    )
    args_schema: type[BaseModel] = ShopSavvyProductSearchToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SHOPSAVVY_API_KEY"),
        description="The ShopSavvy API key. If not provided, it will be loaded from the SHOPSAVVY_API_KEY environment variable.",
    )
    base_url: str = Field(
        default="https://api.shopsavvy.com/v1",
        description="The base URL for the ShopSavvy API.",
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SHOPSAVVY_API_KEY",
                description="API key for the ShopSavvy Data API. Get one at https://shopsavvy.com/data",
                required=True,
            ),
        ]
    )

    def _run(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """Search for products using the ShopSavvy API.

        Args:
            query: The search query (product name, barcode, UPC, ASIN, or URL).

        Returns:
            A JSON string containing the search results.
        """
        if not self.api_key:
            raise ValueError(
                "ShopSavvy API key is required. Set the SHOPSAVVY_API_KEY environment variable "
                "or pass api_key when initializing the tool. Get a key at https://shopsavvy.com/data"
            )

        response = requests.get(
            f"{self.base_url}/products/search",
            params={"query": query},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()

        return json.dumps(response.json(), indent=2)
