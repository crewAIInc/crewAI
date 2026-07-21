"""Spraay Batch Payment Tool for CrewAI.

Enables AI agents to validate, estimate, and execute batch cryptocurrency
payments on Base and 15+ supported chains via the Spraay x402 payment gateway.
"""

import json
from typing import Any, ClassVar

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError
import requests

from crewai_tools.tools.spraay_tool.spraay_x402 import post_with_x402


class SpraayRecipient(BaseModel):
    """A single recipient in a batch payment."""

    address: str = Field(..., description="Recipient wallet address.")
    amount: str = Field(
        ...,
        description="Token amount to send, as a string (e.g. '10.0').",
    )


class SpraayBatchPaymentInput(BaseModel):
    """Input schema for the SpraayBatchPaymentTool."""

    action: str = Field(
        ...,
        description=(
            "The batch payment action to perform. "
            "Options: 'validate' (check recipient list), "
            "'estimate' (get gas and fee estimates), "
            "'execute' (send the batch payment)."
        ),
    )
    token_address: str = Field(
        ...,
        description=(
            "The ERC-20 token contract address to send, "
            "or '0x0000000000000000000000000000000000000000' for native ETH."
        ),
    )
    recipients: list[SpraayRecipient] = Field(
        ...,
        description=(
            "List of payment recipients. Each entry must have "
            "'address' (wallet address) and 'amount' (token amount as string). "
            "Example: [{'address': '0xAbc...', 'amount': '10.0'}]"
        ),
    )
    chain_id: int = Field(
        default=8453,
        description="Chain ID for the payment. Default: 8453 (Base).",
    )
    sender_address: str | None = Field(
        default=None,
        description=(
            "Sender wallet address. Required for 'estimate' and 'execute' actions."
        ),
    )


class SpraayBatchPaymentTool(BaseTool):
    """Tool for batch cryptocurrency payments via the Spraay x402 gateway.

    Send payments to up to 200 recipients in a single transaction with
    ~80% gas savings compared to individual transfers. Supports ERC-20
    tokens and native ETH on Base, Ethereum, Solana, and 13+ other chains.

    Use cases include payroll, grant distributions, DAO disbursements,
    airdrops, and bounty payouts.

    The gateway uses the x402 payment protocol. Validation and estimation
    are free. Execution is paid per request via x402 micropayment — no API
    key or signup required, but a funded wallet private key must be set in
    the SPRAAY_WALLET_PRIVATE_KEY environment variable. Without it, the
    'execute' action returns the gateway's payment requirements instead of
    executing.

    Attributes:
        gateway_url: Base URL of the Spraay gateway.
    """

    name: str = "Spraay Batch Payment"
    description: str = (
        "Validate, estimate gas costs for, and execute batch cryptocurrency "
        "payments to multiple recipients in a single transaction. Supports "
        "ERC-20 tokens and native ETH on Base, Ethereum, Solana, and 13+ chains. "
        "Use 'validate' to check a recipient list, 'estimate' to preview fees, "
        "and 'execute' to send the payment. No API key required — the gateway "
        "uses the x402 payment protocol."
    )
    args_schema: type[BaseModel] = SpraayBatchPaymentInput

    gateway_url: str = "https://gateway.spraay.app"
    _supported_actions: ClassVar[set[str]] = {"validate", "estimate", "execute"}

    def _run(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "").lower()
        if action not in self._supported_actions:
            return (
                f"Error: Invalid action '{action}'. "
                f"Must be one of: {', '.join(sorted(self._supported_actions))}"
            )

        token_address = kwargs.get("token_address", "")
        chain_id = kwargs.get("chain_id", 8453)
        sender_address = kwargs.get("sender_address")

        try:
            recipients = [
                r
                if isinstance(r, SpraayRecipient)
                else SpraayRecipient.model_validate(r)
                for r in kwargs.get("recipients", [])
            ]
        except ValidationError as e:
            return f"Error: Invalid recipient entry: {e}"

        if not recipients:
            return "Error: 'recipients' list is required and cannot be empty."

        if action in ("estimate", "execute") and not sender_address:
            return f"Error: 'sender_address' is required for '{action}' action."

        payload = {
            "tokenAddress": token_address,
            "recipients": [r.model_dump() for r in recipients],
            "chainId": chain_id,
        }
        if sender_address:
            payload["senderAddress"] = sender_address

        if action == "validate":
            return self._validate_batch(payload)
        if action == "estimate":
            return self._estimate_batch(payload)
        return self._execute_batch(payload)

    def _validate_batch(self, payload: dict) -> str:
        """Validate a batch payment recipient list (free endpoint)."""
        try:
            response = requests.post(
                f"{self.gateway_url}/free/validate-batch",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return json.dumps(
                {
                    "status": "valid",
                    "recipientCount": len(payload["recipients"]),
                    "details": data,
                },
                indent=2,
            )
        except requests.RequestException as e:
            return f"Error validating batch: {e}"

    def _estimate_batch(self, payload: dict) -> str:
        """Estimate gas and fees for a batch payment (free endpoint)."""
        try:
            params = {
                "tokenAddress": payload["tokenAddress"],
                "chainId": payload["chainId"],
                "recipientCount": len(payload["recipients"]),
            }
            if "senderAddress" in payload:
                params["senderAddress"] = payload["senderAddress"]
            response = requests.get(
                f"{self.gateway_url}/free/estimate-batch",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return json.dumps(
                {
                    "status": "estimated",
                    "recipientCount": len(payload["recipients"]),
                    "estimate": data,
                },
                indent=2,
            )
        except requests.RequestException as e:
            return f"Error estimating batch: {e}"

    def _execute_batch(self, payload: dict) -> str:
        """Execute a batch payment (x402 paid endpoint)."""
        try:
            paid, data = post_with_x402(
                f"{self.gateway_url}/api/v1/batch/execute",
                payload,
                timeout=60,
            )
            if not paid:
                return json.dumps(data, indent=2)
            return json.dumps(
                {
                    "status": "executed",
                    "recipientCount": len(payload["recipients"]),
                    "result": data,
                },
                indent=2,
            )
        except requests.RequestException as e:
            return f"Error executing batch payment: {e}"
