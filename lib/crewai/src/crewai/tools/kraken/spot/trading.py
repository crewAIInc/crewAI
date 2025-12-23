"""Kraken Spot Trading Tools - Private endpoints for order management."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Add Order
# =============================================================================
class AddOrderInput(BaseModel):
    """Input schema for AddOrderTool."""

    pair: str = Field(..., description="Asset pair (e.g., 'XBTUSD')")
    type: str = Field(..., description="Order direction: 'buy' or 'sell'")
    ordertype: str = Field(
        ...,
        description="Order type: 'market', 'limit', 'stop-loss', 'take-profit', 'stop-loss-limit', 'take-profit-limit', 'trailing-stop', 'trailing-stop-limit', 'settle-position'",
    )
    volume: str = Field(..., description="Order volume in base currency")
    price: str | None = Field(
        default=None, description="Price (required for limit orders)"
    )
    price2: str | None = Field(
        default=None, description="Secondary price (for stop-loss-limit, etc.)"
    )
    leverage: str | None = Field(
        default=None, description="Leverage amount (e.g., '2:1', '3:1')"
    )
    reduce_only: bool | None = Field(
        default=None, description="If true, order will only reduce existing position"
    )
    validate: bool | None = Field(
        default=None, description="If true, validate inputs only without submitting order"
    )
    userref: int | None = Field(
        default=None, description="User reference ID for the order"
    )
    oflags: str | None = Field(
        default=None,
        description="Order flags: 'post' (post-only), 'fcib' (fee in base), 'fciq' (fee in quote), 'nompp' (no market price protection)",
    )


class AddOrderTool(KrakenBaseTool):
    """Place a new spot order on Kraken."""

    name: str = "kraken_add_order"
    description: str = "Place a new spot order on Kraken. Supports market, limit, stop-loss, take-profit, and other order types."
    args_schema: type[BaseModel] = AddOrderInput

    def _run(
        self,
        pair: str,
        type: str,
        ordertype: str,
        volume: str,
        price: str | None = None,
        price2: str | None = None,
        leverage: str | None = None,
        reduce_only: bool | None = None,
        validate: bool | None = None,
        userref: int | None = None,
        oflags: str | None = None,
    ) -> str:
        """Place a new order on Kraken."""
        data: dict[str, Any] = {
            "pair": pair,
            "type": type,
            "ordertype": ordertype,
            "volume": volume,
        }
        if price:
            data["price"] = price
        if price2:
            data["price2"] = price2
        if leverage:
            data["leverage"] = leverage
        if reduce_only is not None:
            data["reduce_only"] = reduce_only
        if validate is not None:
            data["validate"] = validate
        if userref:
            data["userref"] = userref
        if oflags:
            data["oflags"] = oflags

        result = self._private_request("AddOrder", data)
        return str(result)


# =============================================================================
# Tool 2: Add Order Batch
# =============================================================================
class AddOrderBatchInput(BaseModel):
    """Input schema for AddOrderBatchTool."""

    pair: str = Field(..., description="Asset pair for all orders (e.g., 'XBTUSD')")
    orders: str = Field(
        ...,
        description="JSON array of order objects, each with: type, ordertype, volume, price, etc. Max 15 orders.",
    )
    validate: bool | None = Field(
        default=None, description="If true, validate inputs only"
    )


class AddOrderBatchTool(KrakenBaseTool):
    """Place multiple orders in a single request."""

    name: str = "kraken_add_order_batch"
    description: str = "Place multiple orders (up to 15) in a single request for the same asset pair."
    args_schema: type[BaseModel] = AddOrderBatchInput

    def _run(
        self, pair: str, orders: str, validate: bool | None = None
    ) -> str:
        """Place multiple orders on Kraken."""
        data: dict[str, Any] = {"pair": pair, "orders": orders}
        if validate is not None:
            data["validate"] = validate
        result = self._private_request("AddOrderBatch", data)
        return str(result)


# =============================================================================
# Tool 3: Amend Order
# =============================================================================
class AmendOrderInput(BaseModel):
    """Input schema for AmendOrderTool."""

    txid: str = Field(..., description="Transaction ID of the order to amend")
    pair: str = Field(..., description="Asset pair (e.g., 'XBTUSD')")
    volume: str | None = Field(default=None, description="New order volume")
    price: str | None = Field(default=None, description="New order price")
    price2: str | None = Field(default=None, description="New secondary price")


class AmendOrderTool(KrakenBaseTool):
    """Amend an existing open order."""

    name: str = "kraken_amend_order"
    description: str = "Amend an existing open order by changing its volume or price. Faster than edit as it doesn't cancel/recreate."
    args_schema: type[BaseModel] = AmendOrderInput

    def _run(
        self,
        txid: str,
        pair: str,
        volume: str | None = None,
        price: str | None = None,
        price2: str | None = None,
    ) -> str:
        """Amend an order on Kraken."""
        data: dict[str, Any] = {"txid": txid, "pair": pair}
        if volume:
            data["volume"] = volume
        if price:
            data["price"] = price
        if price2:
            data["price2"] = price2
        result = self._private_request("AmendOrder", data)
        return str(result)


# =============================================================================
# Tool 4: Edit Order
# =============================================================================
class EditOrderInput(BaseModel):
    """Input schema for EditOrderTool."""

    txid: str = Field(..., description="Transaction ID of the order to edit")
    pair: str = Field(..., description="Asset pair (e.g., 'XBTUSD')")
    volume: str | None = Field(default=None, description="New order volume")
    price: str | None = Field(default=None, description="New order price")
    price2: str | None = Field(default=None, description="New secondary price")
    oflags: str | None = Field(default=None, description="New order flags")
    validate: bool | None = Field(
        default=None, description="If true, validate inputs only"
    )


class EditOrderTool(KrakenBaseTool):
    """Edit an existing open order (cancel and replace)."""

    name: str = "kraken_edit_order"
    description: str = "Edit an existing open order. This cancels the original order and creates a new one with updated parameters."
    args_schema: type[BaseModel] = EditOrderInput

    def _run(
        self,
        txid: str,
        pair: str,
        volume: str | None = None,
        price: str | None = None,
        price2: str | None = None,
        oflags: str | None = None,
        validate: bool | None = None,
    ) -> str:
        """Edit an order on Kraken."""
        data: dict[str, Any] = {"txid": txid, "pair": pair}
        if volume:
            data["volume"] = volume
        if price:
            data["price"] = price
        if price2:
            data["price2"] = price2
        if oflags:
            data["oflags"] = oflags
        if validate is not None:
            data["validate"] = validate
        result = self._private_request("EditOrder", data)
        return str(result)


# =============================================================================
# Tool 5: Cancel Order
# =============================================================================
class CancelOrderInput(BaseModel):
    """Input schema for CancelOrderTool."""

    txid: str = Field(
        ...,
        description="Transaction ID of order to cancel (or comma-separated list, or userref with # prefix)",
    )


class CancelOrderTool(KrakenBaseTool):
    """Cancel a specific open order."""

    name: str = "kraken_cancel_order"
    description: str = "Cancel a specific open order by its transaction ID. Can also use userref with # prefix."
    args_schema: type[BaseModel] = CancelOrderInput

    def _run(self, txid: str) -> str:
        """Cancel an order on Kraken."""
        result = self._private_request("CancelOrder", {"txid": txid})
        return str(result)


# =============================================================================
# Tool 6: Cancel Order Batch
# =============================================================================
class CancelOrderBatchInput(BaseModel):
    """Input schema for CancelOrderBatchTool."""

    orders: str = Field(
        ...,
        description="JSON array of transaction IDs to cancel (e.g., '[\"TXID1\", \"TXID2\"]')",
    )


class CancelOrderBatchTool(KrakenBaseTool):
    """Cancel multiple orders in a single request."""

    name: str = "kraken_cancel_order_batch"
    description: str = "Cancel multiple orders in a single request using their transaction IDs."
    args_schema: type[BaseModel] = CancelOrderBatchInput

    def _run(self, orders: str) -> str:
        """Cancel multiple orders on Kraken."""
        result = self._private_request("CancelOrderBatch", {"orders": orders})
        return str(result)


# =============================================================================
# Tool 7: Cancel All Orders
# =============================================================================
class CancelAllOrdersTool(KrakenBaseTool):
    """Cancel all open orders."""

    name: str = "kraken_cancel_all_orders"
    description: str = "Cancel ALL open orders. Use with caution! Returns count of cancelled orders."

    def _run(self) -> str:
        """Cancel all orders on Kraken."""
        result = self._private_request("CancelAll")
        return str(result)


# =============================================================================
# Tool 8: Cancel All Orders After X
# =============================================================================
class CancelAllOrdersAfterXInput(BaseModel):
    """Input schema for CancelAllOrdersAfterXTool."""

    timeout: int = Field(
        ...,
        description="Timeout in seconds after which all orders will be cancelled. Set to 0 to disable.",
    )


class CancelAllOrdersAfterXTool(KrakenBaseTool):
    """Set a dead man's switch to cancel all orders after timeout."""

    name: str = "kraken_cancel_all_orders_after_x"
    description: str = "Set a dead man's switch - cancel all orders after X seconds if not reset. Useful for safety. Set timeout to 0 to disable."
    args_schema: type[BaseModel] = CancelAllOrdersAfterXInput

    def _run(self, timeout: int) -> str:
        """Set dead man's switch on Kraken."""
        result = self._private_request("CancelAllOrdersAfter", {"timeout": timeout})
        return str(result)


# =============================================================================
# Tool 9: Get WebSockets Token
# =============================================================================
class GetWebSocketsTokenTool(KrakenBaseTool):
    """Get authentication token for private WebSocket feeds."""

    name: str = "kraken_get_websockets_token"
    description: str = "Get an authentication token for connecting to Kraken's private WebSocket feeds."

    def _run(self) -> str:
        """Get WebSocket token from Kraken."""
        result = self._private_request("GetWebSocketsToken")
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "AddOrderTool",
    "AddOrderBatchTool",
    "AmendOrderTool",
    "EditOrderTool",
    "CancelOrderTool",
    "CancelOrderBatchTool",
    "CancelAllOrdersTool",
    "CancelAllOrdersAfterXTool",
    "GetWebSocketsTokenTool",
]
