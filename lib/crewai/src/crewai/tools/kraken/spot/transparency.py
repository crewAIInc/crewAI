"""Kraken Spot Transparency Tools - Public endpoints for regulatory transparency data."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Get Pre-Trade Data
# =============================================================================
class GetPreTradeDataInput(BaseModel):
    """Input schema for GetPreTradeDataTool."""

    pair: str = Field(..., description="Asset pair (e.g., 'XBTUSD')")


class GetPreTradeDataTool(KrakenBaseTool):
    """Get pre-trade transparency data."""

    name: str = "kraken_get_pre_trade_data"
    description: str = "Get pre-trade transparency data for regulatory compliance. Includes bid/offer prices and volumes."
    args_schema: type[BaseModel] = GetPreTradeDataInput

    def _run(self, pair: str) -> str:
        """Get pre-trade data from Kraken."""
        result = self._public_request("Transparency/PreTrade", {"pair": pair})
        return str(result)


# =============================================================================
# Tool 2: Get Post-Trade Data
# =============================================================================
class GetPostTradeDataInput(BaseModel):
    """Input schema for GetPostTradeDataTool."""

    pair: str = Field(..., description="Asset pair (e.g., 'XBTUSD')")


class GetPostTradeDataTool(KrakenBaseTool):
    """Get post-trade transparency data."""

    name: str = "kraken_get_post_trade_data"
    description: str = "Get post-trade transparency data for regulatory compliance. Includes recent executed trades."
    args_schema: type[BaseModel] = GetPostTradeDataInput

    def _run(self, pair: str) -> str:
        """Get post-trade data from Kraken."""
        result = self._public_request("Transparency/PostTrade", {"pair": pair})
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "GetPreTradeDataTool",
    "GetPostTradeDataTool",
]
