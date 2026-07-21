"""Spraay Balance Tool for CrewAI.

Enables AI agents to check token balances across supported chains
via the Spraay x402 payment gateway's free endpoints.
"""

import json
from typing import Any

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SpraayBalanceInput(BaseModel):
    """Input schema for the SpraayBalanceTool."""

    wallet_address: str = Field(
        ...,
        description="The wallet address to check the balance of.",
    )
    token_address: str = Field(
        default="0x0000000000000000000000000000000000000000",
        description=(
            "The ERC-20 token contract address to check, "
            "or '0x0000000000000000000000000000000000000000' for native ETH. "
            "Default: native ETH."
        ),
    )
    chain_id: int = Field(
        default=8453,
        description="Chain ID to check. Default: 8453 (Base).",
    )


class SpraayBalanceTool(BaseTool):
    """Tool for checking token balances via the Spraay x402 gateway.

    Query wallet balances across 16 supported chains. This is a free
    endpoint that requires no API key, making it useful for pre-flight
    checks before batch payments or escrow creation.

    Attributes:
        gateway_url: Base URL of the Spraay gateway.
    """

    name: str = "Spraay Balance Check"
    description: str = (
        "Check the token balance of a wallet address on any supported chain. "
        "Free endpoint — no API key required. Useful for verifying sufficient "
        "funds before executing batch payments or creating escrow contracts."
    )
    args_schema: type[BaseModel] = SpraayBalanceInput

    gateway_url: str = "https://gateway.spraay.app"

    def _run(self, **kwargs: Any) -> str:
        wallet_address = kwargs.get("wallet_address", "")
        token_address = kwargs.get(
            "token_address",
            "0x0000000000000000000000000000000000000000",
        )
        chain_id = kwargs.get("chain_id", 8453)

        if not wallet_address:
            return "Error: 'wallet_address' is required."

        params = {
            "walletAddress": wallet_address,
            "tokenAddress": token_address,
            "chainId": chain_id,
        }

        try:
            response = requests.get(
                f"{self.gateway_url}/free/balance",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return json.dumps(
                {
                    "status": "success",
                    "walletAddress": wallet_address,
                    "chainId": chain_id,
                    "balance": data,
                },
                indent=2,
            )
        except requests.RequestException as e:
            return f"Error checking balance: {e}"
