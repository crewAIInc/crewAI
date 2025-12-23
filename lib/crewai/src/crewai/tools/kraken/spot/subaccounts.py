"""Kraken Spot Subaccounts Tools - Private endpoints for subaccount management."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Create Subaccount
# =============================================================================
class CreateSubaccountInput(BaseModel):
    """Input schema for CreateSubaccountTool."""

    username: str = Field(..., description="Username for the new subaccount")
    email: str = Field(..., description="Email address for the new subaccount")


class CreateSubaccountTool(KrakenBaseTool):
    """Create a new subaccount."""

    name: str = "kraken_create_subaccount"
    description: str = "Create a new subaccount under your master account."
    args_schema: type[BaseModel] = CreateSubaccountInput

    def _run(self, username: str, email: str) -> str:
        """Create subaccount on Kraken."""
        result = self._private_request(
            "CreateSubaccount", {"username": username, "email": email}
        )
        return str(result)


# =============================================================================
# Tool 2: Account Transfer
# =============================================================================
class AccountTransferInput(BaseModel):
    """Input schema for AccountTransferTool."""

    asset: str = Field(..., description="Asset to transfer")
    amount: str = Field(..., description="Amount to transfer")
    from_account: str = Field(
        ..., description="Source account (your account or subaccount username)"
    )
    to_account: str = Field(
        ..., description="Destination account (your account or subaccount username)"
    )


class AccountTransferTool(KrakenBaseTool):
    """Transfer funds between master account and subaccounts."""

    name: str = "kraken_account_transfer"
    description: str = "Transfer funds between your master account and subaccounts."
    args_schema: type[BaseModel] = AccountTransferInput

    def _run(
        self, asset: str, amount: str, from_account: str, to_account: str
    ) -> str:
        """Transfer between accounts on Kraken."""
        result = self._private_request(
            "AccountTransfer",
            {
                "asset": asset,
                "amount": amount,
                "from": from_account,
                "to": to_account,
            },
        )
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "CreateSubaccountTool",
    "AccountTransferTool",
]
