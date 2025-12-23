"""Kraken Spot Funding Tools - Private endpoints for deposits and withdrawals."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Get Deposit Methods
# =============================================================================
class GetDepositMethodsInput(BaseModel):
    """Input schema for GetDepositMethodsTool."""

    asset: str = Field(..., description="Asset to get deposit methods for (e.g., 'XBT', 'ETH')")


class GetDepositMethodsTool(KrakenBaseTool):
    """Get available deposit methods for an asset."""

    name: str = "kraken_get_deposit_methods"
    description: str = "Get available deposit methods for an asset including fees, limits, and method names."
    args_schema: type[BaseModel] = GetDepositMethodsInput

    def _run(self, asset: str) -> str:
        """Get deposit methods from Kraken."""
        result = self._private_request("DepositMethods", {"asset": asset})
        return str(result)


# =============================================================================
# Tool 2: Get Deposit Addresses
# =============================================================================
class GetDepositAddressesInput(BaseModel):
    """Input schema for GetDepositAddressesTool."""

    asset: str = Field(..., description="Asset to get deposit addresses for")
    method: str = Field(..., description="Deposit method name from GetDepositMethods")
    new: bool | None = Field(
        default=None, description="Generate a new address if true"
    )


class GetDepositAddressesTool(KrakenBaseTool):
    """Get deposit addresses for an asset."""

    name: str = "kraken_get_deposit_addresses"
    description: str = "Get deposit addresses for an asset. Can optionally generate a new address."
    args_schema: type[BaseModel] = GetDepositAddressesInput

    def _run(self, asset: str, method: str, new: bool | None = None) -> str:
        """Get deposit addresses from Kraken."""
        data: dict[str, Any] = {"asset": asset, "method": method}
        if new is not None:
            data["new"] = new
        result = self._private_request("DepositAddresses", data)
        return str(result)


# =============================================================================
# Tool 3: Get Deposit Status
# =============================================================================
class GetDepositStatusInput(BaseModel):
    """Input schema for GetDepositStatusTool."""

    asset: str | None = Field(default=None, description="Filter by asset")
    method: str | None = Field(default=None, description="Filter by deposit method")


class GetDepositStatusTool(KrakenBaseTool):
    """Get status of recent deposits."""

    name: str = "kraken_get_deposit_status"
    description: str = "Get status of recent deposits including amount, status, timestamp, and transaction info."
    args_schema: type[BaseModel] = GetDepositStatusInput

    def _run(self, asset: str | None = None, method: str | None = None) -> str:
        """Get deposit status from Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if method:
            data["method"] = method
        result = self._private_request("DepositStatus", data)
        return str(result)


# =============================================================================
# Tool 4: Get Withdrawal Methods
# =============================================================================
class GetWithdrawalMethodsInput(BaseModel):
    """Input schema for GetWithdrawalMethodsTool."""

    asset: str | None = Field(default=None, description="Filter by asset")
    network: str | None = Field(default=None, description="Filter by network")


class GetWithdrawalMethodsTool(KrakenBaseTool):
    """Get available withdrawal methods."""

    name: str = "kraken_get_withdrawal_methods"
    description: str = "Get available withdrawal methods including fees, limits, and network information."
    args_schema: type[BaseModel] = GetWithdrawalMethodsInput

    def _run(self, asset: str | None = None, network: str | None = None) -> str:
        """Get withdrawal methods from Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if network:
            data["network"] = network
        result = self._private_request("WithdrawMethods", data)
        return str(result)


# =============================================================================
# Tool 5: Get Withdrawal Addresses
# =============================================================================
class GetWithdrawalAddressesInput(BaseModel):
    """Input schema for GetWithdrawalAddressesTool."""

    asset: str | None = Field(default=None, description="Filter by asset")
    method: str | None = Field(default=None, description="Filter by withdrawal method")


class GetWithdrawalAddressesTool(KrakenBaseTool):
    """Get saved withdrawal addresses."""

    name: str = "kraken_get_withdrawal_addresses"
    description: str = "Get saved withdrawal addresses from your account."
    args_schema: type[BaseModel] = GetWithdrawalAddressesInput

    def _run(self, asset: str | None = None, method: str | None = None) -> str:
        """Get withdrawal addresses from Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if method:
            data["method"] = method
        result = self._private_request("WithdrawAddresses", data)
        return str(result)


# =============================================================================
# Tool 6: Get Withdrawal Info
# =============================================================================
class GetWithdrawalInfoInput(BaseModel):
    """Input schema for GetWithdrawalInfoTool."""

    asset: str = Field(..., description="Asset to withdraw")
    key: str = Field(..., description="Withdrawal key name (from saved addresses)")
    amount: str = Field(..., description="Amount to withdraw")


class GetWithdrawalInfoTool(KrakenBaseTool):
    """Get withdrawal fee and limit information."""

    name: str = "kraken_get_withdrawal_info"
    description: str = "Get withdrawal fee and limit information for a specific withdrawal request."
    args_schema: type[BaseModel] = GetWithdrawalInfoInput

    def _run(self, asset: str, key: str, amount: str) -> str:
        """Get withdrawal info from Kraken."""
        result = self._private_request(
            "WithdrawInfo", {"asset": asset, "key": key, "amount": amount}
        )
        return str(result)


# =============================================================================
# Tool 7: Withdraw Funds
# =============================================================================
class WithdrawFundsInput(BaseModel):
    """Input schema for WithdrawFundsTool."""

    asset: str = Field(..., description="Asset to withdraw")
    key: str = Field(..., description="Withdrawal key name (from saved addresses)")
    amount: str = Field(..., description="Amount to withdraw")
    address: str | None = Field(
        default=None, description="Crypto address (for new addresses not yet saved)"
    )


class WithdrawFundsTool(KrakenBaseTool):
    """Initiate a withdrawal."""

    name: str = "kraken_withdraw_funds"
    description: str = "Initiate a withdrawal. CAUTION: This moves real funds! Use GetWithdrawalInfo first to verify fees."
    args_schema: type[BaseModel] = WithdrawFundsInput

    def _run(
        self, asset: str, key: str, amount: str, address: str | None = None
    ) -> str:
        """Withdraw funds from Kraken."""
        data: dict[str, Any] = {"asset": asset, "key": key, "amount": amount}
        if address:
            data["address"] = address
        result = self._private_request("Withdraw", data)
        return str(result)


# =============================================================================
# Tool 8: Get Withdrawal Status
# =============================================================================
class GetWithdrawalStatusInput(BaseModel):
    """Input schema for GetWithdrawalStatusTool."""

    asset: str | None = Field(default=None, description="Filter by asset")
    method: str | None = Field(default=None, description="Filter by withdrawal method")


class GetWithdrawalStatusTool(KrakenBaseTool):
    """Get status of recent withdrawals."""

    name: str = "kraken_get_withdrawal_status"
    description: str = "Get status of recent withdrawals including amount, status, timestamp, and transaction info."
    args_schema: type[BaseModel] = GetWithdrawalStatusInput

    def _run(self, asset: str | None = None, method: str | None = None) -> str:
        """Get withdrawal status from Kraken."""
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
    """Input schema for WalletTransferTool."""

    asset: str = Field(..., description="Asset to transfer")
    amount: str = Field(..., description="Amount to transfer")
    from_wallet: str = Field(
        ..., description="Source wallet: 'Spot Wallet' or 'Futures Wallet'"
    )
    to_wallet: str = Field(
        ..., description="Destination wallet: 'Spot Wallet' or 'Futures Wallet'"
    )


class WalletTransferTool(KrakenBaseTool):
    """Transfer between Kraken wallets."""

    name: str = "kraken_wallet_transfer"
    description: str = "Transfer funds between Spot Wallet and Futures Wallet on Kraken."
    args_schema: type[BaseModel] = WalletTransferInput

    def _run(self, asset: str, amount: str, from_wallet: str, to_wallet: str) -> str:
        """Transfer between wallets on Kraken."""
        result = self._private_request(
            "WalletTransfer",
            {"asset": asset, "amount": amount, "from": from_wallet, "to": to_wallet},
        )
        return str(result)


# =============================================================================
# Tool 10: Cancel Withdrawal
# =============================================================================
class CancelWithdrawalInput(BaseModel):
    """Input schema for CancelWithdrawalTool."""

    asset: str = Field(..., description="Asset of the withdrawal to cancel")
    refid: str = Field(..., description="Reference ID of the withdrawal to cancel")


class CancelWithdrawalTool(KrakenBaseTool):
    """Cancel a pending withdrawal."""

    name: str = "kraken_cancel_withdrawal"
    description: str = "Cancel a pending withdrawal that hasn't been processed yet."
    args_schema: type[BaseModel] = CancelWithdrawalInput

    def _run(self, asset: str, refid: str) -> str:
        """Cancel withdrawal on Kraken."""
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
