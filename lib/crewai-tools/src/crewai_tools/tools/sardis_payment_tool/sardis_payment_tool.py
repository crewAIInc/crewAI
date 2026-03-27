"""Sardis payment tools for CrewAI.

These tools enable CrewAI agents to make policy-controlled payments
through Sardis non-custodial MPC wallets.

Installation:
    pip install sardis-crewai

Or install the tools directly:
    pip install sardis crewai-tools
"""

from __future__ import annotations

import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class SardisPaymentToolInput(BaseModel):
    """Input for SardisPaymentTool."""

    amount: float = Field(description="Payment amount in USD")
    merchant: str = Field(description="Merchant or recipient identifier")
    purpose: str = Field(default="Payment", description="Reason for payment")


class SardisPaymentTool(BaseTool):
    """Execute policy-controlled payments from a Sardis wallet.

    This tool enables AI agents to make real financial transactions
    with spending policy guardrails. Every payment is checked against
    configurable policies before execution.

    Attributes:
        wallet_id: Wallet ID to use. Defaults to SARDIS_WALLET_ID env var.

    Example:
        ```python
        from crewai_tools import SardisPaymentTool

        tool = SardisPaymentTool()
        result = tool.run(amount=25.0, merchant="openai", purpose="API credits")
        ```
    """

    name: str = "Sardis Payment Tool"
    description: str = (
        "Execute a policy-controlled payment from a Sardis wallet. "
        "Checks spending limits and policies before executing. "
        "Supports USDC payments on Base, Polygon, Ethereum, Arbitrum, and Optimism."
    )
    args_schema: type[BaseModel] = SardisPaymentToolInput

    wallet_id: str | None = None

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SARDIS_API_KEY",
                description="API key for Sardis",
                required=True,
            ),
            EnvVar(
                name="SARDIS_WALLET_ID",
                description="Sardis wallet ID to use for payments",
                required=False,
            ),
        ]
    )

    def __init__(self, wallet_id: str | None = None, **kwargs):
        super().__init__(**kwargs)
        if wallet_id:
            self.wallet_id = wallet_id

    def _run(self, **kwargs: Any) -> str:
        try:
            from sardis import SardisClient
        except ImportError:
            return "Error: sardis package required. Install with: pip install sardis"

        amount = kwargs.get("amount")
        merchant = kwargs.get("merchant")
        purpose = kwargs.get("purpose", "Payment")

        if not amount or not merchant:
            return "Error: 'amount' and 'merchant' are required parameters."

        wallet_id = self.wallet_id or os.getenv("SARDIS_WALLET_ID")
        if not wallet_id:
            return "Error: No wallet ID configured. Set SARDIS_WALLET_ID env var."

        client = SardisClient(api_key=os.getenv("SARDIS_API_KEY"))
        result = client.payments.send(
            wallet_id, to=merchant, amount=amount, purpose=purpose
        )

        if result.success:
            return f"APPROVED: ${amount} to {merchant} (tx: {result.tx_id})"
        return f"BLOCKED by policy: {result.message}"
