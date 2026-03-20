import json
import logging
from typing import Any, Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

X402SEARCH_URL = "https://x402search.xyz/v1/search"


class X402SearchToolSchema(BaseModel):
    """Input for X402SearchTool."""

    query: str = Field(
        ...,
        description=(
            "Natural language query to find API services by capability. "
            "Examples: 'token price', 'crypto market data', 'NFT metadata', "
            "'weather forecast', 'sentiment analysis', 'btc price'"
        ),
    )
    limit: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of results to return (1-10, default 5).",
    )


class X402SearchTool(BaseTool):
    name: str = "Search APIs by capability"
    description: str = (
        "Search for API services and data providers by natural language capability query. "
        "Searches 14,000+ indexed APIs across crypto, DeFi, NFT, weather, finance, AI, and more. "
        "Returns matching services with names, descriptions, and endpoints. "
        "Cost: $0.01 USDC per query via x402 protocol on Base mainnet — paid automatically. "
        "Use this to discover what APIs exist for a given capability before calling them. "
        "Best queries: 'token price' (88 results), 'crypto market data' (10), 'btc price' (8)."
    )
    args_schema: Type[BaseModel] = X402SearchToolSchema
    base_url: str = X402SEARCH_URL

    def _run(self, **kwargs: Any) -> Any:
        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 5)

        if not query or len(query.strip()) < 2:
            return "Please provide a search query of at least 2 characters."

        try:
            response = requests.get(
                self.base_url,
                params={"q": query.strip()},
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

        except requests.Timeout:
            return f"x402search timed out for query '{query}'. Try again or use a broader term."
        except requests.HTTPError as e:
            return f"x402search API error: {e.response.status_code}"
        except requests.RequestException as e:
            return f"x402search connection failed: {e}"

        results = data.get("results", data) if isinstance(data, dict) else data
        if not isinstance(results, list):
            return "Unexpected response format from x402search."

        results = results[:limit]

        if not results:
            return (
                f"No API services found for '{query}'. "
                "Try broader terms — 'crypto' returns 112 results, 'token price' returns 88."
            )

        lines = [f"Found API services for '{query}':\n"]
        for i, r in enumerate(results, 1):
            name = r.get("name") or r.get("api_name", "Unknown")
            desc = r.get("description") or r.get("accepts", "")
            url = r.get("url") or r.get("base_url", "")
            lines.append(f"{i}. {name}")
            if desc:
                lines.append(f"   {desc[:120]}{'...' if len(desc) > 120 else ''}")
            if url:
                lines.append(f"   {url}")
            lines.append("")

        return "\n".join(lines).strip()
