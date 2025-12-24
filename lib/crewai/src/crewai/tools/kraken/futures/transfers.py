"""Kraken Futures Transfers Tools - Private endpoints voor wallet transfers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Wallet Transfer
# =============================================================================
class KrakenFuturesWalletTransferInput(BaseModel):
    """Input schema voor KrakenFuturesWalletTransferTool."""

    amount: str = Field(..., description="Bedrag om over te maken")
    from_account: str = Field(
        ..., description="Bron account type: 'cash' (spot) of 'flex' (futures)"
    )
    to_account: str = Field(
        ..., description="Doel account type: 'cash' (spot) of 'flex' (futures)"
    )
    unit: str = Field(
        default="USD", description="Valuta eenheid (bijv. 'USD', 'BTC', 'ETH')"
    )


class KrakenFuturesWalletTransferTool(KrakenFuturesBaseTool):
    """Transfer fondsen tussen wallets."""

    name: str = "kraken_futures_wallet_transfer"
    description: str = "Maak fondsen over tussen Spot wallet (cash) en Futures wallet (flex)."
    args_schema: type[BaseModel] = KrakenFuturesWalletTransferInput

    def _run(
        self, amount: str, from_account: str, to_account: str, unit: str = "USD"
    ) -> str:
        """Transfer tussen wallets op Kraken Futures."""
        data = {
            "amount": amount,
            "fromAccount": from_account,
            "toAccount": to_account,
            "unit": unit,
        }
        result = self._private_request("transfer", data)
        return str(result)


# =============================================================================
# Tool 2: Subaccount Transfer
# =============================================================================
class KrakenFuturesSubAccountTransferInput(BaseModel):
    """Input schema voor KrakenFuturesSubAccountTransferTool."""

    amount: str = Field(..., description="Bedrag om over te maken")
    from_account: str = Field(..., description="Bron subaccount naam of ID")
    to_account: str = Field(..., description="Doel subaccount naam of ID")
    unit: str = Field(
        default="USD", description="Valuta eenheid (bijv. 'USD', 'BTC')"
    )


class KrakenFuturesSubAccountTransferTool(KrakenFuturesBaseTool):
    """Transfer fondsen tussen subaccounts."""

    name: str = "kraken_futures_subaccount_transfer"
    description: str = "Maak fondsen over tussen subaccounts. Vereist master account rechten."
    args_schema: type[BaseModel] = KrakenFuturesSubAccountTransferInput

    def _run(
        self, amount: str, from_account: str, to_account: str, unit: str = "USD"
    ) -> str:
        """Transfer tussen subaccounts op Kraken Futures."""
        data = {
            "amount": amount,
            "fromUser": from_account,
            "toUser": to_account,
            "unit": unit,
        }
        result = self._private_request("transfer/subaccount", data)
        return str(result)


# =============================================================================
# Tool 3: Opname naar Spot
# =============================================================================
class KrakenFuturesWithdrawToSpotInput(BaseModel):
    """Input schema voor KrakenFuturesWithdrawToSpotTool."""

    amount: str = Field(..., description="Bedrag om op te nemen naar Spot wallet")
    currency: str = Field(
        default="USD", description="Valuta om op te nemen (bijv. 'USD', 'BTC')"
    )


class KrakenFuturesWithdrawToSpotTool(KrakenFuturesBaseTool):
    """Neem fondsen op naar Spot wallet."""

    name: str = "kraken_futures_withdraw_to_spot"
    description: str = "Neem fondsen op van Futures wallet naar Kraken Spot wallet."
    args_schema: type[BaseModel] = KrakenFuturesWithdrawToSpotInput

    def _run(self, amount: str, currency: str = "USD") -> str:
        """Neem op naar Spot wallet van Kraken Futures."""
        data = {
            "amount": amount,
            "currency": currency,
        }
        result = self._private_request("withdrawal", data)
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesWalletTransferTool",
    "KrakenFuturesSubAccountTransferTool",
    "KrakenFuturesWithdrawToSpotTool",
]
