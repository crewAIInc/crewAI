"""Kraken Futures Subaccounts Tools - Private endpoints voor subaccount beheer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Haal Subaccounts Op
# =============================================================================
class KrakenFuturesGetSubaccountsTool(KrakenFuturesBaseTool):
    """Haal alle subaccounts op."""

    name: str = "kraken_futures_get_subaccounts"
    description: str = "Haal lijst op van alle subaccounts onder het master account inclusief hun status en saldo's."

    def _run(self) -> str:
        """Haal subaccounts op van Kraken Futures."""
        result = self._private_request("subaccounts", method="GET")
        return str(result)


# =============================================================================
# Tool 2: Check Trading Status
# =============================================================================
class KrakenFuturesCheckTradingStatusInput(BaseModel):
    """Input schema voor KrakenFuturesCheckTradingStatusTool."""

    subaccount_uid: str = Field(..., description="Subaccount UID om status voor te checken")


class KrakenFuturesCheckTradingStatusTool(KrakenFuturesBaseTool):
    """Check trading status van een subaccount."""

    name: str = "kraken_futures_check_trading_status"
    description: str = "Check of een subaccount trading rechten heeft en wat de huidige status is."
    args_schema: type[BaseModel] = KrakenFuturesCheckTradingStatusInput

    def _run(self, subaccount_uid: str) -> str:
        """Check trading status op Kraken Futures."""
        result = self._private_request(
            f"subaccounts/{subaccount_uid}/tradingstatus", method="GET"
        )
        return str(result)


# =============================================================================
# Tool 3: Update Trading Status
# =============================================================================
class KrakenFuturesUpdateTradingStatusInput(BaseModel):
    """Input schema voor KrakenFuturesUpdateTradingStatusTool."""

    subaccount_uid: str = Field(..., description="Subaccount UID om te updaten")
    trading_enabled: bool = Field(
        ..., description="True om trading in te schakelen, False om uit te schakelen"
    )


class KrakenFuturesUpdateTradingStatusTool(KrakenFuturesBaseTool):
    """Update trading status van een subaccount."""

    name: str = "kraken_futures_update_trading_status"
    description: str = "Schakel trading in of uit voor een subaccount."
    args_schema: type[BaseModel] = KrakenFuturesUpdateTradingStatusInput

    def _run(self, subaccount_uid: str, trading_enabled: bool) -> str:
        """Update trading status op Kraken Futures."""
        data = {"tradingEnabled": str(trading_enabled).lower()}
        result = self._private_request(
            f"subaccounts/{subaccount_uid}/tradingstatus", data
        )
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesGetSubaccountsTool",
    "KrakenFuturesCheckTradingStatusTool",
    "KrakenFuturesUpdateTradingStatusTool",
]
