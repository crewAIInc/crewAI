"""Kraken Spot Marktdata Tools - Publieke endpoints voor marktinformatie."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Haal Servertijd Op
# =============================================================================
class GetServerTimeTool(KrakenBaseTool):
    """Haal Kraken servertijd op."""

    name: str = "kraken_get_server_time"
    description: str = "Haal Kraken servertijd op. Handig om API connectiviteit te controleren en timestamps te synchroniseren."

    def _run(self) -> str:
        """Haal de huidige servertijd op van Kraken."""
        result = self._public_request("Time")
        return str(result)


# =============================================================================
# Tool 2: Haal Systeemstatus Op
# =============================================================================
class GetSystemStatusTool(KrakenBaseTool):
    """Haal Kraken systeemstatus op."""

    name: str = "kraken_get_system_status"
    description: str = "Haal Kraken systeemstatus op (online, onderhoud, cancel_only, post_only)."

    def _run(self) -> str:
        """Haal de huidige systeemstatus op van Kraken."""
        result = self._public_request("SystemStatus")
        return str(result)


# =============================================================================
# Tool 3: Haal Asset Info Op
# =============================================================================
class GetAssetInfoInput(BaseModel):
    """Input schema voor GetAssetInfoTool."""

    asset: str | None = Field(
        default=None,
        description="Komma-gescheiden lijst van assets (bijv. 'XBT,ETH'). Laat leeg voor alle assets.",
    )
    aclass: str | None = Field(
        default=None, description="Asset klasse filter (standaard: currency)"
    )


class GetAssetInfoTool(KrakenBaseTool):
    """Haal informatie op over verhandelbare assets."""

    name: str = "kraken_get_asset_info"
    description: str = "Haal informatie op over verhandelbare assets op Kraken inclusief decimalen, weergave decimalen en asset klasse."
    args_schema: type[BaseModel] = GetAssetInfoInput

    def _run(self, asset: str | None = None, aclass: str | None = None) -> str:
        """Haal asset informatie op van Kraken."""
        params: dict[str, str] = {}
        if asset:
            params["asset"] = asset
        if aclass:
            params["aclass"] = aclass
        result = self._public_request("Assets", params)
        return str(result)


# =============================================================================
# Tool 4: Haal Verhandelbare Asset Paren Op
# =============================================================================
class GetTradableAssetPairsInput(BaseModel):
    """Input schema voor GetTradableAssetPairsTool."""

    pair: str | None = Field(
        default=None,
        description="Komma-gescheiden lijst van paren (bijv. 'XBTUSD,ETHUSD'). Laat leeg voor alle paren.",
    )
    info: str | None = Field(
        default=None,
        description="Info om op te halen: 'info' (alles), 'leverage', 'fees', 'margin'",
    )


class GetTradableAssetPairsTool(KrakenBaseTool):
    """Haal verhandelbare asset paren op."""

    name: str = "kraken_get_tradable_asset_pairs"
    description: str = "Haal verhandelbare asset paren en hun details op inclusief fees, hefboom limieten en margin vereisten."
    args_schema: type[BaseModel] = GetTradableAssetPairsInput

    def _run(self, pair: str | None = None, info: str | None = None) -> str:
        """Haal verhandelbare asset paren op van Kraken."""
        params: dict[str, str] = {}
        if pair:
            params["pair"] = pair
        if info:
            params["info"] = info
        result = self._public_request("AssetPairs", params)
        return str(result)


# =============================================================================
# Tool 5: Haal Ticker Informatie Op
# =============================================================================
class GetTickerInput(BaseModel):
    """Input schema voor GetTickerInformationTool."""

    pair: str = Field(
        ...,
        description="Asset paar/paren om ticker voor op te halen (bijv. 'XBTUSD' of 'XBTUSD,ETHUSD' voor meerdere)",
    )


class GetTickerInformationTool(KrakenBaseTool):
    """Haal huidige ticker informatie op voor asset paren."""

    name: str = "kraken_get_ticker"
    description: str = "Haal huidige ticker info op inclusief vraag/bod prijs, laatste handelsprijs, volume, VWAP, aantal trades, hoog, laag en openingsprijs voor asset paren."
    args_schema: type[BaseModel] = GetTickerInput

    def _run(self, pair: str) -> str:
        """Haal ticker informatie op van Kraken."""
        result = self._public_request("Ticker", {"pair": pair})
        return str(result)


# =============================================================================
# Tool 6: Haal Orderboek Op
# =============================================================================
class GetOrderBookInput(BaseModel):
    """Input schema voor GetOrderBookTool."""

    pair: str = Field(..., description="Asset paar (bijv. 'XBTUSD')")
    count: int | None = Field(
        default=None, description="Maximum aantal asks/bids om terug te geven (1-500)"
    )


class GetOrderBookTool(KrakenBaseTool):
    """Haal huidig orderboek op voor een asset paar."""

    name: str = "kraken_get_order_book"
    description: str = "Haal huidig orderboek (asks en bids) op voor een asset paar. Elke entry bevat prijs, volume en timestamp."
    args_schema: type[BaseModel] = GetOrderBookInput

    def _run(self, pair: str, count: int | None = None) -> str:
        """Haal orderboek op van Kraken."""
        params: dict[str, str | int] = {"pair": pair}
        if count:
            params["count"] = count
        result = self._public_request("Depth", params)
        return str(result)


# =============================================================================
# Tool 7: Haal Recente Trades Op
# =============================================================================
class GetRecentTradesInput(BaseModel):
    """Input schema voor GetRecentTradesTool."""

    pair: str = Field(..., description="Asset paar (bijv. 'XBTUSD')")
    since: str | None = Field(
        default=None, description="Geef trades terug sinds gegeven timestamp (nanoseconden)"
    )
    count: int | None = Field(
        default=None, description="Aantal trades om terug te geven (max 1000)"
    )


class GetRecentTradesTool(KrakenBaseTool):
    """Haal recente publieke trades op voor een asset paar."""

    name: str = "kraken_get_recent_trades"
    description: str = "Haal recente publieke trades op voor een asset paar. Geeft prijs, volume, tijd, koop/verkoop, market/limit en overige info terug."
    args_schema: type[BaseModel] = GetRecentTradesInput

    def _run(
        self, pair: str, since: str | None = None, count: int | None = None
    ) -> str:
        """Haal recente trades op van Kraken."""
        params: dict[str, str | int] = {"pair": pair}
        if since:
            params["since"] = since
        if count:
            params["count"] = count
        result = self._public_request("Trades", params)
        return str(result)


# =============================================================================
# Tool 8: Haal Recente Spreads Op
# =============================================================================
class GetRecentSpreadsInput(BaseModel):
    """Input schema voor GetRecentSpreadsTool."""

    pair: str = Field(..., description="Asset paar (bijv. 'XBTUSD')")
    since: str | None = Field(
        default=None, description="Geef spreads terug sinds gegeven timestamp"
    )


class GetRecentSpreadsTool(KrakenBaseTool):
    """Haal recente spread data op voor een asset paar."""

    name: str = "kraken_get_recent_spreads"
    description: str = "Haal recente spread data (bid/ask) op voor een asset paar. Geeft timestamp, bid prijs en ask prijs terug."
    args_schema: type[BaseModel] = GetRecentSpreadsInput

    def _run(self, pair: str, since: str | None = None) -> str:
        """Haal recente spreads op van Kraken."""
        params: dict[str, str] = {"pair": pair}
        if since:
            params["since"] = since
        result = self._public_request("Spread", params)
        return str(result)


# =============================================================================
# Tool 9: Haal OHLC Data Op
# =============================================================================
class GetOHLCInput(BaseModel):
    """Input schema voor GetOHLCDataTool."""

    pair: str = Field(..., description="Asset paar (bijv. 'XBTUSD')")
    interval: int | None = Field(
        default=None,
        description="Tijdsframe interval in minuten: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600",
    )
    since: str | None = Field(
        default=None, description="Geef OHLC data terug sinds gegeven timestamp"
    )


class GetOHLCDataTool(KrakenBaseTool):
    """Haal OHLC (candlestick) data op voor een asset paar."""

    name: str = "kraken_get_ohlc"
    description: str = "Haal OHLC (Open, High, Low, Close) candlestick data op voor een asset paar. Geeft tijd, open, hoog, laag, close, vwap, volume en count terug."
    args_schema: type[BaseModel] = GetOHLCInput

    def _run(
        self, pair: str, interval: int | None = None, since: str | None = None
    ) -> str:
        """Haal OHLC data op van Kraken."""
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
