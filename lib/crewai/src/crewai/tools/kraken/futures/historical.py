"""Kraken Futures Historical Data Tools - Private endpoints voor historische data."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Haal Fills Op
# =============================================================================
class KrakenFuturesGetFillsInput(BaseModel):
    """Input schema voor KrakenFuturesGetFillsTool."""

    last_fill_time: str | None = Field(
        default=None, description="Haal fills op sinds deze timestamp (ISO 8601)"
    )
    symbol: str | None = Field(
        default=None, description="Filter op symbool"
    )


class KrakenFuturesGetFillsTool(KrakenFuturesBaseTool):
    """Haal trade fills/executions op."""

    name: str = "kraken_futures_get_fills"
    description: str = "Haal je trade fills (executions) op met prijs, grootte, fees en timestamps."
    args_schema: type[BaseModel] = KrakenFuturesGetFillsInput

    def _run(
        self, last_fill_time: str | None = None, symbol: str | None = None
    ) -> str:
        """Haal fills op van Kraken Futures."""
        data: dict[str, Any] = {}
        if last_fill_time:
            data["lastFillTime"] = last_fill_time
        if symbol:
            data["symbol"] = symbol
        result = self._private_request("fills", data, method="GET")
        return str(result)


# =============================================================================
# Tool 2: Haal Funding Rates Op
# =============================================================================
class KrakenFuturesGetFundingRatesInput(BaseModel):
    """Input schema voor KrakenFuturesGetFundingRatesTool."""

    symbol: str = Field(..., description="Futures symbool (bijv. 'PI_XBTUSD')")


class KrakenFuturesGetFundingRatesTool(KrakenFuturesBaseTool):
    """Haal historische funding rates op."""

    name: str = "kraken_futures_get_funding_rates"
    description: str = "Haal historische funding rates op voor een perpetual futures contract."
    args_schema: type[BaseModel] = KrakenFuturesGetFundingRatesInput

    def _run(self, symbol: str) -> str:
        """Haal funding rates op van Kraken Futures."""
        result = self._public_request(f"historicalfundingrates/{symbol}")
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesGetFillsTool",
    "KrakenFuturesGetFundingRatesTool",
]
