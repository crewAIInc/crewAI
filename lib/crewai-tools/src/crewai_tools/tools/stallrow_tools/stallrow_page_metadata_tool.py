"""Tool for fetching a web page's declared metadata via StallRow."""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests

from crewai_tools.tools.stallrow_tools.x402_client import X402Client, X402Error


class StallRowPageMetadataToolInput(BaseModel):
    """Input schema for StallRowPageMetadataTool."""

    url: str = Field(
        ...,
        description="The URL of the web page to fetch metadata for, "
        "e.g. 'https://www.example.com/article'.",
    )


class StallRowPageMetadataTool(BaseTool):
    """Fetch a page's declared metadata for $0.005, paid over x402.

    One call returns the page's title, description, canonical URL,
    OpenGraph and Twitter card fields, document language, favicon, the
    h1/h2 heading outline, and word count — the page's identity and
    shape, before an agent decides whether the full content is worth
    fetching.

    There is no API key or account: the StallRow endpoint is paid
    per call over the x402 protocol (USDC on Base). The payer wallet
    comes from the `STALLROW_WALLET_PRIVATE_KEY` environment variable,
    and a hard per-call cap (`STALLROW_MAX_PRICE_USDC`, default
    $0.005 — exactly the endpoint's price) refuses any higher price
    before anything is signed. With no wallet configured the tool
    fails closed with an explanation instead of paying.
    """

    name: str = "StallRow Page Metadata"
    description: str = (
        "Fetches one web page's declared metadata — title, description, "
        "canonical URL, OpenGraph and Twitter card fields, language, "
        "favicon, h1/h2 heading outline, and word count — as JSON. "
        "Costs $0.005 USDC per call, paid automatically over the x402 "
        "protocol on Base from the wallet in STALLROW_WALLET_PRIVATE_KEY; "
        "a hard per-call price cap (default $0.005) refuses higher prices. "
        "Use it to learn a page's identity and shape before deciding "
        "whether its full content is worth fetching."
    )
    args_schema: type[BaseModel] = StallRowPageMetadataToolInput
    base_url: str = "https://stall-row-production.up.railway.app"
    max_price_usdc: str | None = Field(
        default=None,
        description="Per-call price cap in USDC. Defaults to the "
        "STALLROW_MAX_PRICE_USDC environment variable, then $0.005.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["eth-account"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="STALLROW_WALLET_PRIVATE_KEY",
                description="Private key of the Base wallet that pays the "
                "per-call USDC price over x402. Required to run the tool; "
                "without it the tool fails closed and pays nothing.",
                required=False,
            ),
            EnvVar(
                name="STALLROW_MAX_PRICE_USDC",
                description="Hard per-call price cap in USDC. Any endpoint "
                "price above the cap is refused before payment.",
                required=False,
                default="0.005",
            ),
        ]
    )

    def _run(self, url: str, **kwargs: Any) -> str:
        client = X402Client.from_env(max_price_usdc=self.max_price_usdc)
        try:
            result = client.post_json(f"{self.base_url}/v1/page-metadata", {"url": url})
        except X402Error as e:
            return f"Error fetching page metadata: {e!s}"
        except requests.RequestException as e:
            return f"Error fetching page metadata: {e!s}"
        return json.dumps(result)
