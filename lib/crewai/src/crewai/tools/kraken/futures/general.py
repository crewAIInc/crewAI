"""Kraken Futures General Tools - Private endpoints voor algemene functionaliteit."""

from __future__ import annotations

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Haal Notificaties Op
# =============================================================================
class KrakenFuturesGetNotificationsTool(KrakenFuturesBaseTool):
    """Haal account notificaties op."""

    name: str = "kraken_futures_get_notifications"
    description: str = "Haal account notificaties op inclusief margin calls, liquidatie waarschuwingen en andere belangrijke berichten."

    def _run(self) -> str:
        """Haal notificaties op van Kraken Futures."""
        result = self._private_request("notifications", method="GET")
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesGetNotificationsTool",
]
