"""Spraay Escrow Tool for CrewAI.

Enables AI agents to create and manage on-chain escrow contracts
for trustless transactions via the Spraay x402 payment gateway.
"""

import json
from typing import Any, Optional

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SpraayEscrowInput(BaseModel):
    """Input schema for the SpraayEscrowTool."""

    token_address: str = Field(
        ...,
        description=(
            "The ERC-20 token contract address for the escrow, "
            "or '0x0000000000000000000000000000000000000000' for native ETH."
        ),
    )
    amount: str = Field(
        ...,
        description="The amount to escrow (as a string, e.g. '100.0').",
    )
    depositor: str = Field(
        ...,
        description="Wallet address of the party depositing funds.",
    )
    beneficiary: str = Field(
        ...,
        description="Wallet address of the party who will receive funds on release.",
    )
    chain_id: int = Field(
        default=8453,
        description="Chain ID for the escrow. Default: 8453 (Base).",
    )
    conditions: Optional[str] = Field(
        default=None,
        description=(
            "Optional human-readable description of release conditions "
            "for the escrow."
        ),
    )


class SpraayEscrowTool(BaseTool):
    """Tool for creating on-chain escrow contracts via the Spraay x402 gateway.

    Create trustless escrow agreements between two parties with
    programmable release conditions. Funds are held in a smart contract
    until conditions are met, eliminating counterparty risk.

    Use cases include freelance payments, agent-to-agent service contracts,
    milestone-based disbursements, and dispute-protected purchases.

    The gateway uses the x402 payment protocol — no API key or signup
    required.

    Attributes:
        gateway_url: Base URL of the Spraay gateway.
    """

    name: str = "Spraay Escrow"
    description: str = (
        "Create an on-chain escrow contract between two parties. "
        "Funds are locked in a smart contract and released when "
        "conditions are met. Supports ERC-20 tokens and native ETH "
        "on Base, Ethereum, and 14+ other chains. No API key required "
        "— the gateway uses the x402 payment protocol."
    )
    args_schema: type[BaseModel] = SpraayEscrowInput

    gateway_url: str = "https://gateway.spraay.app"

    def _run(self, **kwargs: Any) -> str:
        token_address = kwargs.get("token_address", "")
        amount = kwargs.get("amount", "")
        depositor = kwargs.get("depositor", "")
        beneficiary = kwargs.get("beneficiary", "")
        chain_id = kwargs.get("chain_id", 8453)
        conditions = kwargs.get("conditions")

        if not all([token_address, amount, depositor, beneficiary]):
            return (
                "Error: 'token_address', 'amount', 'depositor', and "
                "'beneficiary' are all required."
            )

        payload = {
            "tokenAddress": token_address,
            "amount": amount,
            "depositor": depositor,
            "beneficiary": beneficiary,
            "chainId": chain_id,
        }
        if conditions:
            payload["conditions"] = conditions

        try:
            response = requests.post(
                f"{self.gateway_url}/api/v1/escrow/create",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return json.dumps(
                {
                    "status": "escrow_created",
                    "depositor": depositor,
                    "beneficiary": beneficiary,
                    "amount": amount,
                    "result": data,
                },
                indent=2,
            )
        except requests.RequestException as e:
            return f"Error creating escrow: {e}"
