"""Kraken Futures Market Data Tools - Publieke endpoints voor marktinformatie."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Haal Alle Tickers Op
# =============================================================================
class KrakenFuturesGetTickersTool(KrakenFuturesBaseTool):
    """Haal tickers op voor alle Futures instrumenten."""

    name: str = "kraken_futures_get_tickers"
    description: str = "Haal tickers op voor alle Futures instrumenten inclusief laatste prijs, bid/ask, volume en 24u statistieken."

    def _run(self) -> str:
        """Haal alle tickers op van Kraken Futures."""
        result = self._public_request("tickers")
        return str(result)


# =============================================================================
# Tool 2: Haal Specifieke Ticker Op
# =============================================================================
class KrakenFuturesGetTickerInput(BaseModel):
    """Input schema voor KrakenFuturesGetTickerTool."""

    symbol: str = Field(..., description="Futures symbool (bijv. 'PI_XBTUSD', 'PF_ETHUSD')")


class KrakenFuturesGetTickerTool(KrakenFuturesBaseTool):
    """Haal ticker op voor een specifiek Futures instrument."""

    name: str = "kraken_futures_get_ticker"
    description: str = "Haal ticker op voor een specifiek Futures instrument inclusief laatste prijs, bid/ask, volume en open interest."
    args_schema: type[BaseModel] = KrakenFuturesGetTickerInput

    def _run(self, symbol: str) -> str:
        """Haal ticker op voor specifiek symbool van Kraken Futures."""
        result = self._public_request(f"tickers/{symbol}")
        return str(result)


# =============================================================================
# Tool 3: Haal Orderboek Op
# =============================================================================
class KrakenFuturesGetOrderBookInput(BaseModel):
    """Input schema voor KrakenFuturesGetOrderBookTool."""

    symbol: str = Field(..., description="Futures symbool (bijv. 'PI_XBTUSD')")


class KrakenFuturesGetOrderBookTool(KrakenFuturesBaseTool):
    """Haal orderboek op voor een Futures instrument."""

    name: str = "kraken_futures_get_order_book"
    description: str = "Haal orderboek (asks en bids) op voor een Futures instrument met prijs en volume voor elk niveau."
    args_schema: type[BaseModel] = KrakenFuturesGetOrderBookInput

    def _run(self, symbol: str) -> str:
        """Haal orderboek op van Kraken Futures."""
        result = self._public_request(f"orderbook/{symbol}")
        return str(result)


# =============================================================================
# Tool 4: Haal Handelsgeschiedenis Op
# =============================================================================
class KrakenFuturesGetTradeHistoryInput(BaseModel):
    """Input schema voor KrakenFuturesGetTradeHistoryTool."""

    symbol: str = Field(..., description="Futures symbool (bijv. 'PI_XBTUSD')")
    last_time: str | None = Field(
        default=None, description="Haal trades op sinds deze timestamp (ISO 8601 formaat)"
    )


class KrakenFuturesGetTradeHistoryTool(KrakenFuturesBaseTool):
    """Haal publieke handelsgeschiedenis op voor een Futures instrument."""

    name: str = "kraken_futures_get_trade_history"
    description: str = "Haal recente publieke trades op voor een Futures instrument met prijs, volume, richting en timestamp."
    args_schema: type[BaseModel] = KrakenFuturesGetTradeHistoryInput

    def _run(self, symbol: str, last_time: str | None = None) -> str:
        """Haal handelsgeschiedenis op van Kraken Futures."""
        params = {}
        if last_time:
            params["lastTime"] = last_time
        result = self._public_request(f"history/{symbol}", params)
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesGetTickersTool",
    "KrakenFuturesGetTickerTool",
    "KrakenFuturesGetOrderBookTool",
    "KrakenFuturesGetTradeHistoryTool",
]
