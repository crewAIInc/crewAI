"""Kraken Spot Market Data Tools - Public endpoints for market information."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Get Server Time
# =============================================================================
class GetServerTimeTool(KrakenBaseTool):
    """Get Kraken server time."""

    name: str = "kraken_get_server_time"
    description: str = "Get Kraken server time. Useful to check API connectivity and synchronize timestamps."

    def _run(self) -> str:
        """Get the current server time from Kraken."""
        result = self._public_request("Time")
        return str(result)


# =============================================================================
# Tool 2: Get System Status
# =============================================================================
class GetSystemStatusTool(KrakenBaseTool):
    """Get Kraken system status."""

    name: str = "kraken_get_system_status"
    description: str = "Get Kraken system status (online, maintenance, cancel_only, post_only)."

    def _run(self) -> str:
        """Get the current system status from Kraken."""
        result = self._public_request("SystemStatus")
        return str(result)


# =============================================================================
# Tool 3: Get Asset Info
# =============================================================================
class GetAssetInfoInput(BaseModel):
    """Input schema for GetAssetInfoTool."""

    asset: str | None = Field(
        default=None,
        description="Comma-separated list of assets (e.g., 'XBT,ETH'). Leave empty for all assets.",
    )
    aclass: str | None = Field(
        default=None, description="Asset class filter (default: currency)"
    )


class GetAssetInfoTool(KrakenBaseTool):
    """Get information about tradeable assets."""

    name: str = "kraken_get_asset_info"
    description: str = "Get information about tradeable assets on Kraken including decimals, display decimals, and asset class."
    args_schema: type[BaseModel] = GetAssetInfoInput

    def _run(self, asset: str | None = None, aclass: str | None = None) -> str:
        """Get asset information from Kraken."""
        params: dict[str, str] = {}
        if asset:
            params["asset"] = asset
        if aclass:
            params["aclass"] = aclass
        result = self._public_request("Assets", params)
        return str(result)


# =============================================================================
# Tool 4: Get Tradable Asset Pairs
# =============================================================================
class GetTradableAssetPairsInput(BaseModel):
    """Input schema for GetTradableAssetPairsTool."""

    pair: str | None = Field(
        default=None,
        description="Comma-separated list of pairs (e.g., 'XBTUSD,ETHUSD'). Leave empty for all pairs.",
    )
    info: str | None = Field(
        default=None,
        description="Info to retrieve: 'info' (all), 'leverage', 'fees', 'margin'",
    )


class GetTradableAssetPairsTool(KrakenBaseTool):
    """Get tradeable asset pairs."""

    name: str = "kraken_get_tradable_asset_pairs"
    description: str = "Get tradeable asset pairs and their details including fees, leverage limits, and margin requirements."
    args_schema: type[BaseModel] = GetTradableAssetPairsInput

    def _run(self, pair: str | None = None, info: str | None = None) -> str:
        """Get tradeable asset pairs from Kraken."""
        params: dict[str, str] = {}
        if pair:
            params["pair"] = pair
        if info:
            params["info"] = info
        result = self._public_request("AssetPairs", params)
        return str(result)


# =============================================================================
# Tool 5: Get Ticker Information
# =============================================================================
class GetTickerInput(BaseModel):
    """Input schema for GetTickerInformationTool."""

    pair: str = Field(
        ...,
        description="Asset pair(s) to get ticker for (e.g., 'XBTUSD' or 'XBTUSD,ETHUSD' for multiple)",
    )


class GetTickerInformationTool(KrakenBaseTool):
    """Get current ticker information for asset pairs."""

    name: str = "kraken_get_ticker"
    description: str = "Get current ticker info including ask/bid price, last trade price, volume, VWAP, number of trades, high, low, and opening price for asset pairs."
    args_schema: type[BaseModel] = GetTickerInput

    def _run(self, pair: str) -> str:
        """Get ticker information from Kraken."""
        result = self._public_request("Ticker", {"pair": pair})
        return str(result)


# =============================================================================
# Tool 6: Get Order Book
# =============================================================================
class GetOrderBookInput(BaseModel):
    """Input schema for GetOrderBookTool."""

    pair: str = Field(..., description="Asset pair (e.g., 'XBTUSD')")
    count: int | None = Field(
        default=None, description="Maximum number of asks/bids to return (1-500)"
    )


class GetOrderBookTool(KrakenBaseTool):
    """Get current order book for an asset pair."""

    name: str = "kraken_get_order_book"
    description: str = "Get current order book (asks and bids) for an asset pair. Each entry contains price, volume, and timestamp."
    args_schema: type[BaseModel] = GetOrderBookInput

    def _run(self, pair: str, count: int | None = None) -> str:
        """Get order book from Kraken."""
        params: dict[str, str | int] = {"pair": pair}
        if count:
            params["count"] = count
        result = self._public_request("Depth", params)
        return str(result)


# =============================================================================
# Tool 7: Get Recent Trades
# =============================================================================
class GetRecentTradesInput(BaseModel):
    """Input schema for GetRecentTradesTool."""

    pair: str = Field(..., description="Asset pair (e.g., 'XBTUSD')")
    since: str | None = Field(
        default=None, description="Return trades since given timestamp (nanoseconds)"
    )
    count: int | None = Field(
        default=None, description="Number of trades to return (max 1000)"
    )


class GetRecentTradesTool(KrakenBaseTool):
    """Get recent public trades for an asset pair."""

    name: str = "kraken_get_recent_trades"
    description: str = "Get recent public trades for an asset pair. Returns price, volume, time, buy/sell, market/limit, and miscellaneous info."
    args_schema: type[BaseModel] = GetRecentTradesInput

    def _run(
        self, pair: str, since: str | None = None, count: int | None = None
    ) -> str:
        """Get recent trades from Kraken."""
        params: dict[str, str | int] = {"pair": pair}
        if since:
            params["since"] = since
        if count:
            params["count"] = count
        result = self._public_request("Trades", params)
        return str(result)


# =============================================================================
# Tool 8: Get Recent Spreads
# =============================================================================
class GetRecentSpreadsInput(BaseModel):
    """Input schema for GetRecentSpreadsTool."""

    pair: str = Field(..., description="Asset pair (e.g., 'XBTUSD')")
    since: str | None = Field(
        default=None, description="Return spreads since given timestamp"
    )


class GetRecentSpreadsTool(KrakenBaseTool):
    """Get recent spread data for an asset pair."""

    name: str = "kraken_get_recent_spreads"
    description: str = "Get recent spread data (bid/ask) for an asset pair. Returns timestamp, bid price, and ask price."
    args_schema: type[BaseModel] = GetRecentSpreadsInput

    def _run(self, pair: str, since: str | None = None) -> str:
        """Get recent spreads from Kraken."""
        params: dict[str, str] = {"pair": pair}
        if since:
            params["since"] = since
        result = self._public_request("Spread", params)
        return str(result)


# =============================================================================
# Tool 9: Get OHLC Data
# =============================================================================
class GetOHLCInput(BaseModel):
    """Input schema for GetOHLCDataTool."""

    pair: str = Field(..., description="Asset pair (e.g., 'XBTUSD')")
    interval: int | None = Field(
        default=None,
        description="Time frame interval in minutes: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600",
    )
    since: str | None = Field(
        default=None, description="Return OHLC data since given timestamp"
    )


class GetOHLCDataTool(KrakenBaseTool):
    """Get OHLC (candlestick) data for an asset pair."""

    name: str = "kraken_get_ohlc"
    description: str = "Get OHLC (Open, High, Low, Close) candlestick data for an asset pair. Returns time, open, high, low, close, vwap, volume, and count."
    args_schema: type[BaseModel] = GetOHLCInput

    def _run(
        self, pair: str, interval: int | None = None, since: str | None = None
    ) -> str:
        """Get OHLC data from Kraken."""
        params: dict[str, str | int] = {"pair": pair}
        if interval:
            params["interval"] = interval
        if since:
            params["since"] = since
        result = self._public_request("OHLC", params)
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "GetServerTimeTool",
    "GetSystemStatusTool",
    "GetAssetInfoTool",
    "GetTradableAssetPairsTool",
    "GetTickerInformationTool",
    "GetOrderBookTool",
    "GetRecentTradesTool",
    "GetRecentSpreadsTool",
    "GetOHLCDataTool",
]
