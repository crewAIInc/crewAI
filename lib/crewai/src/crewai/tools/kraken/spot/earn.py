"""Kraken Spot Earn Tools - Private endpoints voor staking en earn strategieën."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Lijst Earn Strategieën
# =============================================================================
class ListEarnStrategiesInput(BaseModel):
    """Input schema voor ListEarnStrategiesTool."""

    asset: str | None = Field(
        default=None, description="Filter op asset (bijv. 'ETH', 'DOT')"
    )
    lock_type: str | None = Field(
        default=None,
        description="Filter op lock type: 'flex' (flexibel), 'bonded' (vastgezet), 'instant' (directe bonding/unbonding)",
    )


class ListEarnStrategiesTool(KrakenBaseTool):
    """Lijst beschikbare staking/earn strategieën."""

    name: str = "kraken_list_earn_strategies"
    description: str = "Lijst beschikbare staking en earn strategieën inclusief APY, lock type en minimum allocatie."
    args_schema: type[BaseModel] = ListEarnStrategiesInput

    def _run(self, asset: str | None = None, lock_type: str | None = None) -> str:
        """Lijst earn strategieën van Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if lock_type:
            data["lock_type"] = lock_type
        result = self._private_request("Earn/Strategies", data)
        return str(result)


# =============================================================================
# Tool 2: Lijst Earn Allocaties
# =============================================================================
class ListEarnAllocationsInput(BaseModel):
    """Input schema voor ListEarnAllocationsTool."""

    ascending: bool | None = Field(
        default=None, description="Sorteer oplopend op aanmaaktijd"
    )
    converted_asset: str | None = Field(
        default=None, description="Asset om waardes naar te converteren voor weergave"
    )
    hide_zero_allocations: bool | None = Field(
        default=None, description="Verberg strategieën met nul allocaties"
    )


class ListEarnAllocationsTool(KrakenBaseTool):
    """Haal huidige earn allocaties op."""

    name: str = "kraken_list_earn_allocations"
    description: str = "Haal huidige earn/staking allocaties op met gestaakt bedrag, in afwachting zijnde beloningen en status."
    args_schema: type[BaseModel] = ListEarnAllocationsInput

    def _run(
        self,
        ascending: bool | None = None,
        converted_asset: str | None = None,
        hide_zero_allocations: bool | None = None,
    ) -> str:
        """Lijst earn allocaties van Kraken."""
        data: dict[str, Any] = {}
        if ascending is not None:
            data["ascending"] = ascending
        if converted_asset:
            data["converted_asset"] = converted_asset
        if hide_zero_allocations is not None:
            data["hide_zero_allocations"] = hide_zero_allocations
        result = self._private_request("Earn/Allocations", data)
        return str(result)


# =============================================================================
# Tool 3: Alloceer Earn Fondsen
# =============================================================================
class AllocateEarnFundsInput(BaseModel):
    """Input schema voor AllocateEarnFundsTool."""

    strategy_id: str = Field(..., description="Strategie ID van ListEarnStrategies")
    amount: str = Field(..., description="Bedrag om te alloceren/staken")


class AllocateEarnFundsTool(KrakenBaseTool):
    """Alloceer fondsen naar een earn strategie (stake)."""

    name: str = "kraken_allocate_earn_funds"
    description: str = "Alloceer (stake) fondsen naar een earn strategie. Geeft een referentie ID terug om de allocatie te volgen."
    args_schema: type[BaseModel] = AllocateEarnFundsInput

    def _run(self, strategy_id: str, amount: str) -> str:
        """Alloceer fondsen naar earn strategie op Kraken."""
        result = self._private_request(
            "Earn/Allocate", {"strategy_id": strategy_id, "amount": amount}
        )
        return str(result)


# =============================================================================
# Tool 4: Haal Allocatie Status Op
# =============================================================================
class GetAllocationStatusInput(BaseModel):
    """Input schema voor GetAllocationStatusTool."""

    strategy_id: str = Field(..., description="Strategie ID om allocatie status voor te checken")


class GetAllocationStatusTool(KrakenBaseTool):
    """Haal status op van earn allocatie verzoek."""

    name: str = "kraken_get_allocation_status"
    description: str = "Haal de status op van een in afwachting zijnd earn allocatie (staking) verzoek."
    args_schema: type[BaseModel] = GetAllocationStatusInput

    def _run(self, strategy_id: str) -> str:
        """Haal allocatie status op van Kraken."""
        result = self._private_request(
            "Earn/AllocateStatus", {"strategy_id": strategy_id}
        )
        return str(result)


# =============================================================================
# Tool 5: Dealloceer Earn Fondsen
# =============================================================================
class DeallocateEarnFundsInput(BaseModel):
    """Input schema voor DeallocateEarnFundsTool."""

    strategy_id: str = Field(..., description="Strategie ID om van te dealloceren")
    amount: str = Field(..., description="Bedrag om te dealloceren/unstaken")


class DeallocateEarnFundsTool(KrakenBaseTool):
    """Dealloceer fondsen van earn strategie (unstake)."""

    name: str = "kraken_deallocate_earn_funds"
    description: str = "Dealloceer (unstake) fondsen van een earn strategie. Kan een unbonding periode hebben afhankelijk van strategie."
    args_schema: type[BaseModel] = DeallocateEarnFundsInput

    def _run(self, strategy_id: str, amount: str) -> str:
        """Dealloceer fondsen van earn strategie op Kraken."""
        result = self._private_request(
            "Earn/Deallocate", {"strategy_id": strategy_id, "amount": amount}
        )
        return str(result)


# =============================================================================
# Tool 6: Haal Deallocatie Status Op
# =============================================================================
class GetDeallocationStatusInput(BaseModel):
    """Input schema voor GetDeallocationStatusTool."""

    strategy_id: str = Field(..., description="Strategie ID om deallocatie status voor te checken")


class GetDeallocationStatusTool(KrakenBaseTool):
    """Haal status op van earn deallocatie verzoek."""

    name: str = "kraken_get_deallocation_status"
    description: str = "Haal de status op van een in afwachting zijnd earn deallocatie (unstaking) verzoek."
    args_schema: type[BaseModel] = GetDeallocationStatusInput

    def _run(self, strategy_id: str) -> str:
        """Haal deallocatie status op van Kraken."""
        result = self._private_request(
            "Earn/DeallocateStatus", {"strategy_id": strategy_id}
        )
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "ListEarnStrategiesTool",
    "ListEarnAllocationsTool",
    "AllocateEarnFundsTool",
    "GetAllocationStatusTool",
    "DeallocateEarnFundsTool",
    "GetDeallocationStatusTool",
]
