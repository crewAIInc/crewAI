from typing import Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.scavio_tool.base import ScavioBaseTool


class ScavioAmazonSearchToolSchema(BaseModel):
    """Input schema for ScavioAmazonSearchTool."""

    query: str = Field(..., description="Product search query (e.g., 'wireless headphones').")


class ScavioAmazonSearchTool(ScavioBaseTool):
    """Tool that searches Amazon product listings using the Scavio API.

    Attributes:
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The Scavio API key.
        max_results: The maximum number of results to return.
        domain: The Amazon marketplace to search.
        sort_by: Sort order for results.
    """

    name: str = "Scavio Amazon Search"
    description: str = (
        "A tool that searches Amazon product listings using the Scavio API. "
        "It returns product names, prices, ratings, ASINs, and availability "
        "as a JSON string. Use for product research, price comparisons, "
        "or shopping queries."
    )
    args_schema: type[BaseModel] = ScavioAmazonSearchToolSchema

    domain: str = Field(
        default="com",
        description=(
            "Amazon marketplace to search. Supported values: "
            '"com" (US), "co.uk" (UK), "ca" (Canada), "de" (Germany), '
            '"fr" (France), "co.jp" (Japan), "in" (India), etc.'
        ),
    )
    sort_by: Literal[
        "most_recent",
        "price_low_to_high",
        "price_high_to_low",
        "featured",
        "average_review",
        "bestsellers",
    ] | None = Field(
        default=None,
        description="Sort order for results.",
    )

    def _run(self, query: str) -> str:
        """Synchronously searches Amazon product listings.

        Args:
            query: Product search query.

        Returns:
            A JSON string containing product search results.
        """
        if not self.client:
            raise ValueError(
                "Scavio client is not initialized. Ensure 'scavio' is "
                "installed and API key is set."
            )

        raw = self.client.amazon.search(
            query=query,
            domain=self.domain,
            sort_by=self.sort_by,
        )

        data = raw.get("data")
        if isinstance(data, dict):
            products = data.get("products")
            if isinstance(products, list) and len(products) > self.max_results:
                raw["data"]["products"] = products[: self.max_results]

        return self._format_response(raw)

    async def _arun(self, query: str) -> str:
        """Asynchronously searches Amazon product listings.

        Args:
            query: Product search query.

        Returns:
            A JSON string containing product search results.
        """
        if not self.async_client:
            raise ValueError(
                "Scavio async client is not initialized. Ensure 'scavio' "
                "is installed and API key is set."
            )

        raw = await self.async_client.amazon.search(
            query=query,
            domain=self.domain,
            sort_by=self.sort_by,
        )

        data = raw.get("data")
        if isinstance(data, dict):
            products = data.get("products")
            if isinstance(products, list) and len(products) > self.max_results:
                raw["data"]["products"] = products[: self.max_results]

        return self._format_response(raw)
