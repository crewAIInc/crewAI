"""Kraken Futures Instruments Tools - Publieke endpoints voor instrument informatie."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Haal Instrumenten Op
# =============================================================================
class KrakenFuturesGetInstrumentsTool(KrakenFuturesBaseTool):
    """Haal alle beschikbare Futures instrumenten op."""

    name: str = "kraken_futures_get_instruments"
    description: str = "Haal lijst op van alle beschikbare Futures instrumenten inclusief contractgrootte, tick size, margin vereisten en limieten."

    def _run(self) -> str:
        """Haal instrumenten op van Kraken Futures."""
        result = self._public_request("instruments")
        return str(result)


# =============================================================================
# Tool 2: Haal Instrument Status Op
# =============================================================================
class KrakenFuturesGetInstrumentStatusInput(BaseModel):
    """Input schema voor KrakenFuturesGetInstrumentStatusTool."""

    symbol: str = Field(..., description="Futures symbool (bijv. 'PI_XBTUSD')")


class KrakenFuturesGetInstrumentStatusTool(KrakenFuturesBaseTool):
    """Haal status op van een specifiek instrument."""

    name: str = "kraken_futures_get_instrument_status"
    description: str = "Haal huidige status op van een Futures instrument inclusief trading status, settlement info en expiratie details."
    args_schema: type[BaseModel] = KrakenFuturesGetInstrumentStatusInput

    def _run(self, symbol: str) -> str:
        """Haal instrument status op van Kraken Futures."""
        result = self._public_request(f"instruments/{symbol}")
        return str(result)


# =============================================================================
# Tool 3: Haal Alle Instrument Statussen Op
# =============================================================================
class KrakenFuturesGetInstrumentStatusListTool(KrakenFuturesBaseTool):
    """Haal statussen op van alle instrumenten."""

    name: str = "kraken_futures_get_instrument_status_list"
    description: str = "Haal huidige status op van alle Futures instrumenten in één request."

    def _run(self) -> str:
        """Haal alle instrument statussen op van Kraken Futures."""
        result = self._public_request("instruments/status")
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesGetInstrumentsTool",
    "KrakenFuturesGetInstrumentStatusTool",
    "KrakenFuturesGetInstrumentStatusListTool",
]
