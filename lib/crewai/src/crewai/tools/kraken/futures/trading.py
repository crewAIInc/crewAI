"""Kraken Futures Trading Settings Tools - Private endpoints voor trading instellingen."""

from __future__ import annotations

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Haal Self-Trade Strategie Op
# =============================================================================
class KrakenFuturesGetSelfTradeStrategyTool(KrakenFuturesBaseTool):
    """Haal self-trade preventie strategie op."""

    name: str = "kraken_futures_get_self_trade_strategy"
    description: str = "Haal de huidige self-trade preventie strategie op. Bepaalt hoe orders worden behandeld die tegen je eigen orders zouden handelen."

    def _run(self) -> str:
        """Haal self-trade strategie op van Kraken Futures."""
        result = self._private_request("selftradestrategy", method="GET")
        return str(result)


# =============================================================================
# Tool 2: Stel Self-Trade Strategie In
# =============================================================================
class KrakenFuturesSetSelfTradeStrategyInput(BaseModel):
    """Input schema voor KrakenFuturesSetSelfTradeStrategyTool."""

    strategy: str = Field(
        ...,
        description="Self-trade strategie: 'cancel_resting' (annuleer bestaande), 'cancel_aggressor' (annuleer nieuwe), 'none' (sta toe)",
    )


class KrakenFuturesSetSelfTradeStrategyTool(KrakenFuturesBaseTool):
    """Stel self-trade preventie strategie in."""

    name: str = "kraken_futures_set_self_trade_strategy"
    description: str = "Stel de self-trade preventie strategie in. Bepaalt wat er gebeurt als je orders tegen elkaar zouden handelen."
    args_schema: type[BaseModel] = KrakenFuturesSetSelfTradeStrategyInput

    def _run(self, strategy: str) -> str:
        """Stel self-trade strategie in op Kraken Futures."""
        data = {"strategy": strategy}
        result = self._private_request("selftradestrategy", data)
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesGetSelfTradeStrategyTool",
    "KrakenFuturesSetSelfTradeStrategyTool",
]
