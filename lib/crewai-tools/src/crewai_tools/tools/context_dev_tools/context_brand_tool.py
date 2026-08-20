from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from crewai_tools.tools.context_dev_tools.base import ContextDevBaseTool, compact


class ContextBrandToolSchema(BaseModel):
    identifier: str = Field(
        min_length=1,
        description="Domain, company name, work email, ticker, direct URL, or transaction descriptor.",
    )
    lookup_type: Literal[
        "domain", "name", "email", "ticker", "direct_url", "transaction"
    ] = Field(
        default="domain",
        description="How Context.dev should interpret the identifier.",
    )
    max_speed: bool = Field(
        default=False,
        description="Return faster, less comprehensive brand data when true.",
    )
    country: str | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Country hint for company-name or transaction lookups.",
    )
    exchange: str | None = Field(
        default=None,
        description="Stock exchange for ticker lookups.",
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        le=300000,
        description="Maximum server processing time in milliseconds.",
    )


class ContextBrandTool(ContextDevBaseTool):
    name: str = "Context.dev brand intelligence"
    description: str = (
        "Retrieve structured brand intelligence including logos, colors, company "
        "descriptions, social profiles, links, industry, and location."
    )
    args_schema: type[BaseModel] = ContextBrandToolSchema

    def _run(
        self,
        identifier: str,
        lookup_type: Literal[
            "domain", "name", "email", "ticker", "direct_url", "transaction"
        ] = "domain",
        max_speed: bool = False,
        country: str | None = None,
        exchange: str | None = None,
        timeout_ms: int | None = None,
    ) -> Any:
        identifier_field = {
            "domain": "domain",
            "name": "name",
            "email": "email",
            "ticker": "ticker",
            "direct_url": "direct_url",
            "transaction": "transaction_info",
        }[lookup_type]
        return self._request(
            "POST",
            "/brand/retrieve",
            json_body=compact(
                {
                    "type": f"by_{lookup_type}",
                    identifier_field: identifier,
                    "maxSpeed": max_speed if lookup_type != "direct_url" else None,
                    "country_gl": country
                    if lookup_type in {"name", "transaction"}
                    else None,
                    "ticker_exchange": exchange if lookup_type == "ticker" else None,
                    "timeoutMS": timeout_ms,
                }
            ),
        )
