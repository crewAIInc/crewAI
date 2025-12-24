"""Kraken Futures Fee Schedules Tools - Private endpoints voor fee informatie."""

from __future__ import annotations

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Haal Fee Schedules Op
# =============================================================================
class KrakenFuturesGetFeeSchedulesTool(KrakenFuturesBaseTool):
    """Haal fee schedules op."""

    name: str = "kraken_futures_get_fee_schedules"
    description: str = "Haal fee schedules op inclusief maker/taker fees per volume tier en huidige tier."

    def _run(self) -> str:
        """Haal fee schedules op van Kraken Futures."""
        result = self._private_request("feeschedules", method="GET")
        return str(result)


# =============================================================================
# Tool 2: Haal Fee Volumes Op
# =============================================================================
class KrakenFuturesGetFeeVolumesTool(KrakenFuturesBaseTool):
    """Haal fee volumes op."""

    name: str = "kraken_futures_get_fee_volumes"
    description: str = "Haal je 30-dagen handelsvolume op dat gebruikt wordt voor fee tier berekening."

    def _run(self) -> str:
        """Haal fee volumes op van Kraken Futures."""
        result = self._private_request("feeschedules/volumes", method="GET")
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesGetFeeSchedulesTool",
    "KrakenFuturesGetFeeVolumesTool",
]
