"""CrewAI tools for cross-chain DeFi operations via the Suwappu DEX API.

Provides token pricing, swap quotes, portfolio tracking, and chain/token
discovery across 15+ blockchain networks including Ethereum, Base, Arbitrum,
Solana, and more.

Dependencies:
    - suwappu
    - pydantic
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


def _get_client():
    """Create a Suwappu API client using the configured API key."""
    try:
        from suwappu import create_client
    except ImportError:
        raise ImportError(
            "The 'suwappu' package is required for SuwappuDefiTools. "
            "Install it with: pip install suwappu"
        )
    api_key = os.environ.get("SUWAPPU_API_KEY", "")
    return create_client(api_key=api_key)


def _run_async(coro):
    """Run an async coroutine in a sync context."""
    try:
        asyncio.get_running_loop()
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


# --- Schemas ---


class GetPricesSchema(BaseModel):
    """Input for SuwappuGetPricesTool."""

    token: str = Field(..., description="Token symbol or address to look up (e.g. 'ETH', 'USDC')")
    chain: str = Field(default="", description="Optional chain to filter by (e.g. 'base', 'ethereum')")


class GetQuoteSchema(BaseModel):
    """Input for SuwappuGetQuoteTool."""

    from_token: str = Field(..., description="Token symbol to sell (e.g. 'ETH')")
    to_token: str = Field(..., description="Token symbol to buy (e.g. 'USDC')")
    amount: float = Field(..., description="Amount of from_token to swap")
    chain: str = Field(..., description="Blockchain network for the swap (e.g. 'base')")


class GetPortfolioSchema(BaseModel):
    """Input for SuwappuGetPortfolioTool."""

    chain: str = Field(default="", description="Optional chain to filter portfolio by")


class ListTokensSchema(BaseModel):
    """Input for SuwappuListTokensTool."""

    chain: str = Field(..., description="Chain to list tokens for (e.g. 'base', 'ethereum')")


# --- Tools ---


class SuwappuGetPricesTool(BaseTool):
    """Tool for getting real-time token prices and 24h changes across chains.

    Returns current USD price and 24-hour price change for any token
    supported by the Suwappu DEX aggregator.
    """

    name: str = "Suwappu Get Token Prices"
    description: str = (
        "Get the current USD price and 24-hour change for a cryptocurrency token. "
        "Optionally filter by blockchain network."
    )
    args_schema: Type[BaseModel] = GetPricesSchema
    env_vars: List[EnvVar] = [
        EnvVar(name="SUWAPPU_API_KEY", description="API key for Suwappu DEX", required=True),
    ]

    def _run(self, **kwargs: Any) -> str:
        token = kwargs.get("token", "")
        chain = kwargs.get("chain", "")

        async def _fetch():
            client = _get_client()
            try:
                result = await client.get_prices(token, chain or None)
                return result.model_dump() if hasattr(result, "model_dump") else str(result)
            finally:
                await client.close()

        return str(_run_async(_fetch()))


class SuwappuGetQuoteTool(BaseTool):
    """Tool for getting cross-chain swap quotes with price, route, gas, and fees.

    Returns a detailed quote for swapping one token to another, including
    the best execution route, estimated gas costs, and protocol fees.
    """

    name: str = "Suwappu Get Swap Quote"
    description: str = (
        "Get a swap quote for trading one token for another. "
        "Returns price impact, route, estimated gas, and fees. "
        "Use this before executing a swap to preview the trade."
    )
    args_schema: Type[BaseModel] = GetQuoteSchema
    env_vars: List[EnvVar] = [
        EnvVar(name="SUWAPPU_API_KEY", description="API key for Suwappu DEX", required=True),
    ]

    def _run(self, **kwargs: Any) -> str:
        from_token = kwargs.get("from_token", "")
        to_token = kwargs.get("to_token", "")
        amount = kwargs.get("amount", 0)
        chain = kwargs.get("chain", "")

        async def _fetch():
            client = _get_client()
            try:
                result = await client.get_quote(from_token, to_token, amount, chain)
                return result.model_dump() if hasattr(result, "model_dump") else str(result)
            finally:
                await client.close()

        return str(_run_async(_fetch()))


class SuwappuGetPortfolioTool(BaseTool):
    """Tool for checking wallet token balances across all supported chains.

    Returns the current token holdings and balances for the connected
    wallet, optionally filtered to a specific chain.
    """

    name: str = "Suwappu Get Portfolio"
    description: str = (
        "Check wallet token balances across all supported blockchain networks, "
        "or filter by a specific chain."
    )
    args_schema: Type[BaseModel] = GetPortfolioSchema
    env_vars: List[EnvVar] = [
        EnvVar(name="SUWAPPU_API_KEY", description="API key for Suwappu DEX", required=True),
    ]

    def _run(self, **kwargs: Any) -> str:
        chain = kwargs.get("chain", "")

        async def _fetch():
            client = _get_client()
            try:
                result = await client.get_portfolio(chain or None)
                return [
                    r.model_dump() if hasattr(r, "model_dump") else str(r)
                    for r in result
                ]
            finally:
                await client.close()

        return str(_run_async(_fetch()))


class SuwappuListChainsTool(BaseTool):
    """Tool for listing all blockchain networks supported by Suwappu.

    Returns the full list of supported chains (e.g. Ethereum, Base,
    Arbitrum, Solana, etc.) with their chain IDs and metadata.
    """

    name: str = "Suwappu List Supported Chains"
    description: str = "List all blockchain networks supported by the Suwappu DEX aggregator."
    env_vars: List[EnvVar] = [
        EnvVar(name="SUWAPPU_API_KEY", description="API key for Suwappu DEX", required=True),
    ]

    def _run(self, **kwargs: Any) -> str:
        async def _fetch():
            client = _get_client()
            try:
                result = await client.list_chains()
                return [
                    r.model_dump() if hasattr(r, "model_dump") else str(r)
                    for r in result
                ]
            finally:
                await client.close()

        return str(_run_async(_fetch()))


class SuwappuListTokensTool(BaseTool):
    """Tool for listing available tokens on a specific blockchain.

    Returns all tradeable tokens on the given chain, including their
    symbols, addresses, and decimals.
    """

    name: str = "Suwappu List Tokens"
    description: str = "List all available tokens on a specific blockchain network."
    args_schema: Type[BaseModel] = ListTokensSchema
    env_vars: List[EnvVar] = [
        EnvVar(name="SUWAPPU_API_KEY", description="API key for Suwappu DEX", required=True),
    ]

    def _run(self, **kwargs: Any) -> str:
        chain = kwargs.get("chain", "")

        async def _fetch():
            client = _get_client()
            try:
                result = await client.list_tokens(chain)
                return [
                    r.model_dump() if hasattr(r, "model_dump") else str(r)
                    for r in result
                ]
            finally:
                await client.close()

        return str(_run_async(_fetch()))
