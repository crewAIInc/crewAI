"""Kraken Futures Multi-Collateral Tools - Private endpoints voor hefboom en PnL valuta instellingen."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Haal Hefboom Instellingen Op
# =============================================================================
class KrakenFuturesGetLeverageInput(BaseModel):
    """Input schema voor KrakenFuturesGetLeverageTool."""

    symbol: str | None = Field(
        default=None, description="Futures symbool om hefboom voor op te halen (optioneel)"
    )


class KrakenFuturesGetLeverageTool(KrakenFuturesBaseTool):
    """Haal hefboom instellingen op."""

    name: str = "kraken_futures_get_leverage"
    description: str = "Haal huidige hefboom instellingen op voor je account of specifiek symbool. Toont max hefboom en huidige instelling."
    args_schema: type[BaseModel] = KrakenFuturesGetLeverageInput

    def _run(self, symbol: str | None = None) -> str:
        """Haal hefboom op van Kraken Futures."""
        data: dict[str, Any] = {}
        if symbol:
            data["symbol"] = symbol
        result = self._private_request("leveragepreferences", data, method="GET")
        return str(result)


# =============================================================================
# Tool 2: Stel Hefboom In
# =============================================================================
class KrakenFuturesSetLeverageInput(BaseModel):
    """Input schema voor KrakenFuturesSetLeverageTool."""

    symbol: str = Field(..., description="Futures symbool om hefboom voor in te stellen")
    max_leverage: str = Field(..., description="Gewenste maximale hefboom (bijv. '5', '10', '20')")


class KrakenFuturesSetLeverageTool(KrakenFuturesBaseTool):
    """Stel hefboom in voor een symbool."""

    name: str = "kraken_futures_set_leverage"
    description: str = "Stel de maximale hefboom in voor een specifiek Futures symbool. Beïnvloedt initial margin vereisten."
    args_schema: type[BaseModel] = KrakenFuturesSetLeverageInput

    def _run(self, symbol: str, max_leverage: str) -> str:
        """Stel hefboom in op Kraken Futures."""
        data = {
            "symbol": symbol,
            "maxLeverage": max_leverage,
        }
        result = self._private_request("leveragepreferences", data)
        return str(result)


# =============================================================================
# Tool 3: Haal PnL Valuta Voorkeur Op
# =============================================================================
class KrakenFuturesGetPnLCurrencyInput(BaseModel):
    """Input schema voor KrakenFuturesGetPnLCurrencyTool."""

    symbol: str | None = Field(
        default=None, description="Futures symbool (optioneel)"
    )


class KrakenFuturesGetPnLCurrencyTool(KrakenFuturesBaseTool):
    """Haal PnL valuta voorkeur op."""

    name: str = "kraken_futures_get_pnl_currency"
    description: str = "Haal de voorkeurs valuta op waarin PnL wordt afgerekend (bijv. USD, EUR, of de onderliggende crypto)."
    args_schema: type[BaseModel] = KrakenFuturesGetPnLCurrencyInput

    def _run(self, symbol: str | None = None) -> str:
        """Haal PnL valuta op van Kraken Futures."""
        data: dict[str, Any] = {}
        if symbol:
            data["symbol"] = symbol
        result = self._private_request("pnlpreferences", data, method="GET")
        return str(result)


# =============================================================================
# Tool 4: Stel PnL Valuta Voorkeur In
# =============================================================================
class KrakenFuturesSetPnLCurrencyInput(BaseModel):
    """Input schema voor KrakenFuturesSetPnLCurrencyTool."""

    symbol: str = Field(..., description="Futures symbool")
    pnl_currency: str = Field(
        ..., description="Voorkeurs PnL valuta (bijv. 'USD', 'EUR', 'BTC')"
    )


class KrakenFuturesSetPnLCurrencyTool(KrakenFuturesBaseTool):
    """Stel PnL valuta voorkeur in."""

    name: str = "kraken_futures_set_pnl_currency"
    description: str = "Stel de voorkeurs valuta in waarin PnL wordt afgerekend voor een specifiek symbool."
    args_schema: type[BaseModel] = KrakenFuturesSetPnLCurrencyInput

    def _run(self, symbol: str, pnl_currency: str) -> str:
        """Stel PnL valuta in op Kraken Futures."""
        data = {
            "symbol": symbol,
            "pnlCurrency": pnl_currency,
        }
        result = self._private_request("pnlpreferences", data)
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesGetLeverageTool",
    "KrakenFuturesSetLeverageTool",
    "KrakenFuturesGetPnLCurrencyTool",
    "KrakenFuturesSetPnLCurrencyTool",
]
