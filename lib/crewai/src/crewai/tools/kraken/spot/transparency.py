"""Kraken Spot Transparency Tools - Publieke endpoints voor regulatoire transparantie data."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Haal Pre-Trade Data Op
# =============================================================================
class GetPreTradeDataInput(BaseModel):
    """Input schema voor GetPreTradeDataTool."""

    pair: str = Field(..., description="Asset paar (bijv. 'XBTUSD')")


class GetPreTradeDataTool(KrakenBaseTool):
    """Haal pre-trade transparantie data op."""

    name: str = "kraken_get_pre_trade_data"
    description: str = "Haal pre-trade transparantie data op voor regulatoire compliance. Bevat vraag/bod prijzen en volumes."
    args_schema: type[BaseModel] = GetPreTradeDataInput

    def _run(self, pair: str) -> str:
        """Haal pre-trade data op van Kraken."""
        result = self._public_request("Transparency/PreTrade", {"pair": pair})
        return str(result)


# =============================================================================
# Tool 2: Haal Post-Trade Data Op
# =============================================================================
class GetPostTradeDataInput(BaseModel):
    """Input schema voor GetPostTradeDataTool."""

    pair: str = Field(..., description="Asset paar (bijv. 'XBTUSD')")


class GetPostTradeDataTool(KrakenBaseTool):
    """Haal post-trade transparantie data op."""

    name: str = "kraken_get_post_trade_data"
    description: str = "Haal post-trade transparantie data op voor regulatoire compliance. Bevat recente uitgevoerde trades."
    args_schema: type[BaseModel] = GetPostTradeDataInput

    def _run(self, pair: str) -> str:
        """Haal post-trade data op van Kraken."""
        result = self._public_request("Transparency/PostTrade", {"pair": pair})
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "GetPreTradeDataTool",
    "GetPostTradeDataTool",
]
