import json
from typing import Any, Literal

from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from crewai_tools.tools.decodo_base_tool.decodo_base_tool import DecodoBaseTool


__all__ = ["DecodoEcommerceArgs", "DecodoEcommerceTool"]

Store = Literal["amazon", "walmart", "target"]


class DecodoEcommerceArgs(BaseModel):
    store: Store = Field(
        description="Store to look up: 'amazon', 'walmart' or 'target'"
    )
    product_id: str | None = Field(
        default=None,
        description=(
            "Get one product's details. Amazon: the ASIN, e.g. 'B09H74FXNW'. "
            "Walmart or Target: the product ID from the product URL, e.g. '15296401808'"
        ),
    )
    query: str | None = Field(
        default=None, description="Search the store's products for this query instead"
    )
    zip_code: str | None = Field(
        default=None,
        description="US ZIP code to see prices and availability for that delivery location, e.g. '10001'",
    )

    @model_validator(mode="after")
    def _one_of_product_id_or_query(self) -> Self:
        if bool(self.product_id) == bool(self.query):
            raise ValueError("Pass either product_id or query, not both.")
        return self


class DecodoEcommerceTool(DecodoBaseTool):
    """Get product details or search results from Amazon, Walmart or Target, parsed as JSON.

    Args:
        api_key: Decodo Web Scraping API key. Defaults to the ``DECODO_API_KEY`` environment variable.
    """

    name: str = "Decodo Ecommerce"
    description: str = (
        "Look up products on Amazon, Walmart or Target and return structured JSON. "
        "Pass product_id for one product's details: title, price, rating, seller, availability. "
        "Pass query to search the store and get matching products with prices."
    )
    args_schema: type[BaseModel] = DecodoEcommerceArgs

    def _run(
        self,
        store: Store,
        product_id: str | None = None,
        query: str | None = None,
        zip_code: str | None = None,
    ) -> str | ToolFailure:
        from decodo import (
            AmazonProductParams,
            AmazonSearchParams,
            TargetProductParams,
            TargetSearchParams,
            WalmartProductParams,
            WalmartSearchParams,
        )

        if bool(product_id) == bool(query):
            return ToolFailure(
                message="Pass either product_id or query, not both.",
                code="invalid_params",
            )

        zip_code = zip_code or None
        params: Any
        if store == "amazon":
            params = (
                AmazonProductParams(query=product_id, geo=zip_code, parse=True)
                if product_id
                else AmazonSearchParams(query=query, geo=zip_code, parse=True)
            )
        elif store == "walmart":
            params = (
                WalmartProductParams(
                    product_id=product_id, delivery_zip=zip_code, parse=True
                )
                if product_id
                else WalmartSearchParams(query=query, delivery_zip=zip_code, parse=True)
            )
        else:
            # Target pages need JavaScript rendering to load product data.
            params = (
                TargetProductParams(
                    product_id=product_id,
                    delivery_zip=zip_code,
                    headless="html",
                    parse=True,
                )
                if product_id
                else TargetSearchParams(
                    query=query, delivery_zip=zip_code, headless="html", parse=True
                )
            )

        content = self._scrape(params)
        if isinstance(content, ToolFailure | str):
            return content
        return json.dumps(content)
