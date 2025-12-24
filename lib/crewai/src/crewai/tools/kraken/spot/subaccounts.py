"""Kraken Spot Subaccounts Tools - Private endpoints voor subaccount beheer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Maak Subaccount Aan
# =============================================================================
class CreateSubaccountInput(BaseModel):
    """Input schema voor CreateSubaccountTool."""

    username: str = Field(..., description="Gebruikersnaam voor het nieuwe subaccount")
    email: str = Field(..., description="E-mailadres voor het nieuwe subaccount")


class CreateSubaccountTool(KrakenBaseTool):
    """Maak een nieuw subaccount aan."""

    name: str = "kraken_create_subaccount"
    description: str = "Maak een nieuw subaccount aan onder je master account."
    args_schema: type[BaseModel] = CreateSubaccountInput

    def _run(self, username: str, email: str) -> str:
        """Maak subaccount aan op Kraken."""
        result = self._private_request(
            "CreateSubaccount", {"username": username, "email": email}
        )
        return str(result)


# =============================================================================
# Tool 2: Account Transfer
# =============================================================================
class AccountTransferInput(BaseModel):
    """Input schema voor AccountTransferTool."""

    asset: str = Field(..., description="Asset om over te maken")
    amount: str = Field(..., description="Bedrag om over te maken")
    from_account: str = Field(
        ..., description="Bron account (je account of subaccount gebruikersnaam)"
    )
    to_account: str = Field(
        ..., description="Bestemming account (je account of subaccount gebruikersnaam)"
    )


class AccountTransferTool(KrakenBaseTool):
    """Maak fondsen over tussen master account en subaccounts."""

    name: str = "kraken_account_transfer"
    description: str = "Maak fondsen over tussen je master account en subaccounts."
    args_schema: type[BaseModel] = AccountTransferInput

    def _run(
        self, asset: str, amount: str, from_account: str, to_account: str
    ) -> str:
        """Transfer tussen accounts op Kraken."""
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
