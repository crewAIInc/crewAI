"""Spraay Escrow Tool for CrewAI.

Enables AI agents to create and manage on-chain escrow contracts
for trustless transactions via the Spraay x402 payment gateway.
"""

import json
from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests

from crewai_tools.tools.spraay_tool.spraay_payload import (
    chain_slug,
    to_base_units,
    token_decimals,
)
from crewai_tools.tools.spraay_tool.spraay_x402 import post_with_x402


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
    conditions: str | None = Field(
        default=None,
        description=(
            "Optional human-readable description of release conditions for the escrow."
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
    required. Escrow creation is a paid endpoint: set a funded wallet
    private key in the SPRAAY_WALLET_PRIVATE_KEY environment variable
    to pay the x402 micropayment automatically. Without it, the tool
    returns the gateway's payment requirements instead of creating
    the escrow.

    Attributes:
        gateway_url: Base URL of the Spraay gateway.
    """

    name: str = "Spraay Escrow"
    description: str = (
        "Create an on-chain escrow contract between two parties. "
        "Funds are locked in a smart contract and released when "
        "conditions are met. Supports ERC-20 tokens and native coins "
        "on nine EVM chains (Base, Ethereum, BNB Chain, Unichain, "
        "Polygon, Plasma, Arbitrum, Avalanche, BOB). No API key required "
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

        try:
            chain = chain_slug(chain_id)
            base_amount = to_base_units(amount, token_decimals(token_address, chain_id))
        except ValueError as e:
            return f"Error: {e}"

        # The gateway expects {depositor, beneficiary, token, amount} with
        # the amount in base units (per the gateway OpenAPI spec). The chain
        # slug and conditions are extra hints the gateway tolerates.
        payload = {
            "depositor": depositor,
            "beneficiary": beneficiary,
            "token": token_address,
            "amount": base_amount,
            "chain": chain,
        }
        if conditions:
            payload["conditions"] = conditions

        try:
            paid, data = post_with_x402(
                f"{self.gateway_url}/api/v1/escrow/create",
                payload,
                timeout=60,
            )
            if not paid:
                return json.dumps(data, indent=2)
            return json.dumps(
                {
                    "status": "escrow_created",
                    "depositor": depositor,
                    "beneficiary": beneficiary,
                    "amount": amount,
                    "amountBaseUnits": base_amount,
                    "result": data,
                },
                indent=2,
            )
        except requests.RequestException as e:
            return f"Error creating escrow: {e}"
