"""Kraken Spot Earn Tools - Private endpoints for staking and earn strategies."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: List Earn Strategies
# =============================================================================
class ListEarnStrategiesInput(BaseModel):
    """Input schema for ListEarnStrategiesTool."""

    asset: str | None = Field(
        default=None, description="Filter by asset (e.g., 'ETH', 'DOT')"
    )
    lock_type: str | None = Field(
        default=None,
        description="Filter by lock type: 'flex' (flexible), 'bonded' (locked), 'instant' (instant bonding/unbonding)",
    )


class ListEarnStrategiesTool(KrakenBaseTool):
    """List available staking/earn strategies."""

    name: str = "kraken_list_earn_strategies"
    description: str = "List available staking and earn strategies including APY, lock type, and minimum allocation."
    args_schema: type[BaseModel] = ListEarnStrategiesInput

    def _run(self, asset: str | None = None, lock_type: str | None = None) -> str:
        """List earn strategies from Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if lock_type:
            data["lock_type"] = lock_type
        result = self._private_request("Earn/Strategies", data)
        return str(result)


# =============================================================================
# Tool 2: List Earn Allocations
# =============================================================================
class ListEarnAllocationsInput(BaseModel):
    """Input schema for ListEarnAllocationsTool."""

    ascending: bool | None = Field(
        default=None, description="Sort ascending by creation time"
    )
    converted_asset: str | None = Field(
        default=None, description="Asset to convert values to for display"
    )
    hide_zero_allocations: bool | None = Field(
        default=None, description="Hide strategies with zero allocations"
    )


class ListEarnAllocationsTool(KrakenBaseTool):
    """Get current earn allocations."""

    name: str = "kraken_list_earn_allocations"
    description: str = "Get current earn/staking allocations showing amount staked, pending rewards, and status."
    args_schema: type[BaseModel] = ListEarnAllocationsInput

    def _run(
        self,
        ascending: bool | None = None,
        converted_asset: str | None = None,
        hide_zero_allocations: bool | None = None,
    ) -> str:
        """List earn allocations from Kraken."""
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
# Tool 3: Allocate Earn Funds
# =============================================================================
class AllocateEarnFundsInput(BaseModel):
    """Input schema for AllocateEarnFundsTool."""

    strategy_id: str = Field(..., description="Strategy ID from ListEarnStrategies")
    amount: str = Field(..., description="Amount to allocate/stake")


class AllocateEarnFundsTool(KrakenBaseTool):
    """Allocate funds to an earn strategy (stake)."""

    name: str = "kraken_allocate_earn_funds"
    description: str = "Allocate (stake) funds to an earn strategy. Returns a reference ID to track the allocation."
    args_schema: type[BaseModel] = AllocateEarnFundsInput

    def _run(self, strategy_id: str, amount: str) -> str:
        """Allocate funds to earn strategy on Kraken."""
        result = self._private_request(
            "Earn/Allocate", {"strategy_id": strategy_id, "amount": amount}
        )
        return str(result)


# =============================================================================
# Tool 4: Get Allocation Status
# =============================================================================
class GetAllocationStatusInput(BaseModel):
    """Input schema for GetAllocationStatusTool."""

    strategy_id: str = Field(..., description="Strategy ID to check allocation status for")


class GetAllocationStatusTool(KrakenBaseTool):
    """Get status of earn allocation request."""

    name: str = "kraken_get_allocation_status"
    description: str = "Get the status of a pending earn allocation (staking) request."
    args_schema: type[BaseModel] = GetAllocationStatusInput

    def _run(self, strategy_id: str) -> str:
        """Get allocation status from Kraken."""
        result = self._private_request(
            "Earn/AllocateStatus", {"strategy_id": strategy_id}
        )
        return str(result)


# =============================================================================
# Tool 5: Deallocate Earn Funds
# =============================================================================
class DeallocateEarnFundsInput(BaseModel):
    """Input schema for DeallocateEarnFundsTool."""

    strategy_id: str = Field(..., description="Strategy ID to deallocate from")
    amount: str = Field(..., description="Amount to deallocate/unstake")


class DeallocateEarnFundsTool(KrakenBaseTool):
    """Deallocate funds from earn strategy (unstake)."""

    name: str = "kraken_deallocate_earn_funds"
    description: str = "Deallocate (unstake) funds from an earn strategy. May have unbonding period depending on strategy."
    args_schema: type[BaseModel] = DeallocateEarnFundsInput

    def _run(self, strategy_id: str, amount: str) -> str:
        """Deallocate funds from earn strategy on Kraken."""
        result = self._private_request(
            "Earn/Deallocate", {"strategy_id": strategy_id, "amount": amount}
        )
        return str(result)


# =============================================================================
# Tool 6: Get Deallocation Status
# =============================================================================
class GetDeallocationStatusInput(BaseModel):
    """Input schema for GetDeallocationStatusTool."""

    strategy_id: str = Field(..., description="Strategy ID to check deallocation status for")


class GetDeallocationStatusTool(KrakenBaseTool):
    """Get status of earn deallocation request."""

    name: str = "kraken_get_deallocation_status"
    description: str = "Get the status of a pending earn deallocation (unstaking) request."
    args_schema: type[BaseModel] = GetDeallocationStatusInput

    def _run(self, strategy_id: str) -> str:
        """Get deallocation status from Kraken."""
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
