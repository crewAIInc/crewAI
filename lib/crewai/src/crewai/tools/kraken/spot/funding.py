"""Kraken Spot Funding Tools - Private endpoints voor stortingen en opnames."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Haal Stortingsmethodes Op
# =============================================================================
class GetDepositMethodsInput(BaseModel):
    """Input schema voor GetDepositMethodsTool."""

    asset: str = Field(..., description="Asset om stortingsmethodes voor op te halen (bijv. 'XBT', 'ETH')")


class GetDepositMethodsTool(KrakenBaseTool):
    """Haal beschikbare stortingsmethodes op voor een asset."""

    name: str = "kraken_get_deposit_methods"
    description: str = "Haal beschikbare stortingsmethodes op voor een asset inclusief fees, limieten en methode namen."
    args_schema: type[BaseModel] = GetDepositMethodsInput

    def _run(self, asset: str) -> str:
        """Haal stortingsmethodes op van Kraken."""
        result = self._private_request("DepositMethods", {"asset": asset})
        return str(result)


# =============================================================================
# Tool 2: Haal Stortingsadressen Op
# =============================================================================
class GetDepositAddressesInput(BaseModel):
    """Input schema voor GetDepositAddressesTool."""

    asset: str = Field(..., description="Asset om stortingsadressen voor op te halen")
    method: str = Field(..., description="Stortingsmethode naam van GetDepositMethods")
    new: bool | None = Field(
        default=None, description="Genereer een nieuw adres indien true"
    )


class GetDepositAddressesTool(KrakenBaseTool):
    """Haal stortingsadressen op voor een asset."""

    name: str = "kraken_get_deposit_addresses"
    description: str = "Haal stortingsadressen op voor een asset. Kan optioneel een nieuw adres genereren."
    args_schema: type[BaseModel] = GetDepositAddressesInput

    def _run(self, asset: str, method: str, new: bool | None = None) -> str:
        """Haal stortingsadressen op van Kraken."""
        data: dict[str, Any] = {"asset": asset, "method": method}
        if new is not None:
            data["new"] = new
        result = self._private_request("DepositAddresses", data)
        return str(result)


# =============================================================================
# Tool 3: Haal Stortingsstatus Op
# =============================================================================
class GetDepositStatusInput(BaseModel):
    """Input schema voor GetDepositStatusTool."""

    asset: str | None = Field(default=None, description="Filter op asset")
    method: str | None = Field(default=None, description="Filter op stortingsmethode")


class GetDepositStatusTool(KrakenBaseTool):
    """Haal status op van recente stortingen."""

    name: str = "kraken_get_deposit_status"
    description: str = "Haal status op van recente stortingen inclusief bedrag, status, timestamp en transactie info."
    args_schema: type[BaseModel] = GetDepositStatusInput

    def _run(self, asset: str | None = None, method: str | None = None) -> str:
        """Haal stortingsstatus op van Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if method:
            data["method"] = method
        result = self._private_request("DepositStatus", data)
        return str(result)


# =============================================================================
# Tool 4: Haal Opnamemethodes Op
# =============================================================================
class GetWithdrawalMethodsInput(BaseModel):
    """Input schema voor GetWithdrawalMethodsTool."""

    asset: str | None = Field(default=None, description="Filter op asset")
    network: str | None = Field(default=None, description="Filter op netwerk")


class GetWithdrawalMethodsTool(KrakenBaseTool):
    """Haal beschikbare opnamemethodes op."""

    name: str = "kraken_get_withdrawal_methods"
    description: str = "Haal beschikbare opnamemethodes op inclusief fees, limieten en netwerk informatie."
    args_schema: type[BaseModel] = GetWithdrawalMethodsInput

    def _run(self, asset: str | None = None, network: str | None = None) -> str:
        """Haal opnamemethodes op van Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if network:
            data["network"] = network
        result = self._private_request("WithdrawMethods", data)
        return str(result)


# =============================================================================
# Tool 5: Haal Opnameadressen Op
# =============================================================================
class GetWithdrawalAddressesInput(BaseModel):
    """Input schema voor GetWithdrawalAddressesTool."""

    asset: str | None = Field(default=None, description="Filter op asset")
    method: str | None = Field(default=None, description="Filter op opnamemethode")


class GetWithdrawalAddressesTool(KrakenBaseTool):
    """Haal opgeslagen opnameadressen op."""

    name: str = "kraken_get_withdrawal_addresses"
    description: str = "Haal opgeslagen opnameadressen op van je account."
    args_schema: type[BaseModel] = GetWithdrawalAddressesInput

    def _run(self, asset: str | None = None, method: str | None = None) -> str:
        """Haal opnameadressen op van Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if method:
            data["method"] = method
        result = self._private_request("WithdrawAddresses", data)
        return str(result)


# =============================================================================
# Tool 6: Haal Opname Info Op
# =============================================================================
class GetWithdrawalInfoInput(BaseModel):
    """Input schema voor GetWithdrawalInfoTool."""

    asset: str = Field(..., description="Asset om op te nemen")
    key: str = Field(..., description="Opname sleutel naam (van opgeslagen adressen)")
    amount: str = Field(..., description="Bedrag om op te nemen")


class GetWithdrawalInfoTool(KrakenBaseTool):
    """Haal opname fee en limiet informatie op."""

    name: str = "kraken_get_withdrawal_info"
    description: str = "Haal opname fee en limiet informatie op voor een specifiek opname verzoek."
    args_schema: type[BaseModel] = GetWithdrawalInfoInput

    def _run(self, asset: str, key: str, amount: str) -> str:
        """Haal opname info op van Kraken."""
        result = self._private_request(
            "WithdrawInfo", {"asset": asset, "key": key, "amount": amount}
        )
        return str(result)


# =============================================================================
# Tool 7: Neem Fondsen Op
# =============================================================================
class WithdrawFundsInput(BaseModel):
    """Input schema voor WithdrawFundsTool."""

    asset: str = Field(..., description="Asset om op te nemen")
    key: str = Field(..., description="Opname sleutel naam (van opgeslagen adressen)")
    amount: str = Field(..., description="Bedrag om op te nemen")
    address: str | None = Field(
        default=None, description="Crypto adres (voor nieuwe adressen nog niet opgeslagen)"
    )


class WithdrawFundsTool(KrakenBaseTool):
    """Start een opname."""

    name: str = "kraken_withdraw_funds"
    description: str = "Start een opname. LET OP: Dit verplaatst echte fondsen! Gebruik GetWithdrawalInfo eerst om fees te verifiëren."
    args_schema: type[BaseModel] = WithdrawFundsInput

    def _run(
        self, asset: str, key: str, amount: str, address: str | None = None
    ) -> str:
        """Neem fondsen op van Kraken."""
        data: dict[str, Any] = {"asset": asset, "key": key, "amount": amount}
        if address:
            data["address"] = address
        result = self._private_request("Withdraw", data)
        return str(result)


# =============================================================================
# Tool 8: Haal Opnamestatus Op
# =============================================================================
class GetWithdrawalStatusInput(BaseModel):
    """Input schema voor GetWithdrawalStatusTool."""

    asset: str | None = Field(default=None, description="Filter op asset")
    method: str | None = Field(default=None, description="Filter op opnamemethode")


class GetWithdrawalStatusTool(KrakenBaseTool):
    """Haal status op van recente opnames."""

    name: str = "kraken_get_withdrawal_status"
    description: str = "Haal status op van recente opnames inclusief bedrag, status, timestamp en transactie info."
    args_schema: type[BaseModel] = GetWithdrawalStatusInput

    def _run(self, asset: str | None = None, method: str | None = None) -> str:
        """Haal opnamestatus op van Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if method:
            data["method"] = method
        result = self._private_request("WithdrawStatus", data)
        return str(result)


# =============================================================================
# Tool 9: Wallet Transfer
# =============================================================================
class WalletTransferInput(BaseModel):
    """Input schema voor WalletTransferTool."""

    asset: str = Field(..., description="Asset om over te maken")
    amount: str = Field(..., description="Bedrag om over te maken")
    from_wallet: str = Field(
        ..., description="Bron wallet: 'Spot Wallet' of 'Futures Wallet'"
    )
    to_wallet: str = Field(
        ..., description="Bestemming wallet: 'Spot Wallet' of 'Futures Wallet'"
    )


class WalletTransferTool(KrakenBaseTool):
    """Transfer tussen Kraken wallets."""

    name: str = "kraken_wallet_transfer"
    description: str = "Maak fondsen over tussen Spot Wallet en Futures Wallet op Kraken."
    args_schema: type[BaseModel] = WalletTransferInput

    def _run(self, asset: str, amount: str, from_wallet: str, to_wallet: str) -> str:
        """Transfer tussen wallets op Kraken."""
        result = self._private_request(
            "WalletTransfer",
            {"asset": asset, "amount": amount, "from": from_wallet, "to": to_wallet},
        )
        return str(result)


# =============================================================================
# Tool 10: Annuleer Opname
# =============================================================================
class CancelWithdrawalInput(BaseModel):
    """Input schema voor CancelWithdrawalTool."""

    asset: str = Field(..., description="Asset van de opname om te annuleren")
    refid: str = Field(..., description="Referentie ID van de opname om te annuleren")


class CancelWithdrawalTool(KrakenBaseTool):
    """Annuleer een in afwachting zijnde opname."""

    name: str = "kraken_cancel_withdrawal"
    description: str = "Annuleer een in afwachting zijnde opname die nog niet verwerkt is."
    args_schema: type[BaseModel] = CancelWithdrawalInput

    def _run(self, asset: str, refid: str) -> str:
        """Annuleer opname op Kraken."""
        result = self._private_request(
            "WithdrawCancel", {"asset": asset, "refid": refid}
        )
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "GetDepositMethodsTool",
    "GetDepositAddressesTool",
    "GetDepositStatusTool",
    "GetWithdrawalMethodsTool",
    "GetWithdrawalAddressesTool",
    "GetWithdrawalInfoTool",
    "WithdrawFundsTool",
    "GetWithdrawalStatusTool",
    "WalletTransferTool",
    "CancelWithdrawalTool",
]
