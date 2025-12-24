"""Kraken Futures Order Management Tools - Private endpoints voor orderbeheer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool


# =============================================================================
# Tool 1: Plaats Order
# =============================================================================
class KrakenFuturesSendOrderInput(BaseModel):
    """Input schema voor KrakenFuturesSendOrderTool."""

    order_type: str = Field(
        ..., description="Order type: 'lmt' (limit), 'mkt' (market), 'stp' (stop), 'take_profit'"
    )
    symbol: str = Field(..., description="Futures symbool (bijv. 'PI_XBTUSD')")
    side: str = Field(..., description="Order richting: 'buy' of 'sell'")
    size: str = Field(..., description="Order grootte in contracten")
    limit_price: str | None = Field(
        default=None, description="Limiet prijs (vereist voor limit orders)"
    )
    stop_price: str | None = Field(
        default=None, description="Stop prijs (vereist voor stop orders)"
    )
    reduce_only: bool | None = Field(
        default=None, description="Indien true, order kan alleen positie verkleinen"
    )
    client_order_id: str | None = Field(
        default=None, description="Client order ID voor tracking"
    )


class KrakenFuturesSendOrderTool(KrakenFuturesBaseTool):
    """Plaats een nieuwe Futures order."""

    name: str = "kraken_futures_send_order"
    description: str = "Plaats een nieuwe Futures order. Ondersteunt market, limit, stop en take-profit order types."
    args_schema: type[BaseModel] = KrakenFuturesSendOrderInput

    def _run(
        self,
        order_type: str,
        symbol: str,
        side: str,
        size: str,
        limit_price: str | None = None,
        stop_price: str | None = None,
        reduce_only: bool | None = None,
        client_order_id: str | None = None,
    ) -> str:
        """Plaats een order op Kraken Futures."""
        data: dict[str, Any] = {
            "orderType": order_type,
            "symbol": symbol,
            "side": side,
            "size": size,
        }
        if limit_price:
            data["limitPrice"] = limit_price
        if stop_price:
            data["stopPrice"] = stop_price
        if reduce_only is not None:
            data["reduceOnly"] = str(reduce_only).lower()
        if client_order_id:
            data["cliOrdId"] = client_order_id

        result = self._private_request("sendorder", data)
        return str(result)


# =============================================================================
# Tool 2: Wijzig Order
# =============================================================================
class KrakenFuturesEditOrderInput(BaseModel):
    """Input schema voor KrakenFuturesEditOrderTool."""

    order_id: str | None = Field(default=None, description="Order ID om te wijzigen")
    client_order_id: str | None = Field(
        default=None, description="Client order ID om te wijzigen"
    )
    size: str | None = Field(default=None, description="Nieuwe order grootte")
    limit_price: str | None = Field(default=None, description="Nieuwe limiet prijs")
    stop_price: str | None = Field(default=None, description="Nieuwe stop prijs")


class KrakenFuturesEditOrderTool(KrakenFuturesBaseTool):
    """Wijzig een bestaande Futures order."""

    name: str = "kraken_futures_edit_order"
    description: str = "Wijzig een bestaande Futures order. Kan grootte, limiet prijs of stop prijs aanpassen."
    args_schema: type[BaseModel] = KrakenFuturesEditOrderInput

    def _run(
        self,
        order_id: str | None = None,
        client_order_id: str | None = None,
        size: str | None = None,
        limit_price: str | None = None,
        stop_price: str | None = None,
    ) -> str:
        """Wijzig een order op Kraken Futures."""
        data: dict[str, Any] = {}
        if order_id:
            data["orderId"] = order_id
        if client_order_id:
            data["cliOrdId"] = client_order_id
        if size:
            data["size"] = size
        if limit_price:
            data["limitPrice"] = limit_price
        if stop_price:
            data["stopPrice"] = stop_price

        result = self._private_request("editorder", data)
        return str(result)


# =============================================================================
# Tool 3: Annuleer Order
# =============================================================================
class KrakenFuturesCancelOrderInput(BaseModel):
    """Input schema voor KrakenFuturesCancelOrderTool."""

    order_id: str | None = Field(default=None, description="Order ID om te annuleren")
    client_order_id: str | None = Field(
        default=None, description="Client order ID om te annuleren"
    )


class KrakenFuturesCancelOrderTool(KrakenFuturesBaseTool):
    """Annuleer een Futures order."""

    name: str = "kraken_futures_cancel_order"
    description: str = "Annuleer een specifieke Futures order op basis van order ID of client order ID."
    args_schema: type[BaseModel] = KrakenFuturesCancelOrderInput

    def _run(
        self,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> str:
        """Annuleer een order op Kraken Futures."""
        data: dict[str, Any] = {}
        if order_id:
            data["order_id"] = order_id
        if client_order_id:
            data["cliOrdId"] = client_order_id

        result = self._private_request("cancelorder", data)
        return str(result)


# =============================================================================
# Tool 4: Annuleer Alle Orders
# =============================================================================
class KrakenFuturesCancelAllOrdersInput(BaseModel):
    """Input schema voor KrakenFuturesCancelAllOrdersTool."""

    symbol: str | None = Field(
        default=None, description="Optioneel: annuleer alleen orders voor dit symbool"
    )


class KrakenFuturesCancelAllOrdersTool(KrakenFuturesBaseTool):
    """Annuleer alle open Futures orders."""

    name: str = "kraken_futures_cancel_all_orders"
    description: str = "Annuleer alle open Futures orders. Optioneel beperkt tot een specifiek symbool."
    args_schema: type[BaseModel] = KrakenFuturesCancelAllOrdersInput

    def _run(self, symbol: str | None = None) -> str:
        """Annuleer alle orders op Kraken Futures."""
        data: dict[str, Any] = {}
        if symbol:
            data["symbol"] = symbol

        result = self._private_request("cancelallorders", data)
        return str(result)


# =============================================================================
# Tool 5: Batch Order Operaties
# =============================================================================
class KrakenFuturesBatchOrderInput(BaseModel):
    """Input schema voor KrakenFuturesBatchOrderTool."""

    batch_order: str = Field(
        ...,
        description="JSON array van order operaties. Elk object bevat 'order' (send/cancel/edit) en relevante parameters.",
    )


class KrakenFuturesBatchOrderTool(KrakenFuturesBaseTool):
    """Voer batch order operaties uit."""

    name: str = "kraken_futures_batch_order"
    description: str = "Voer meerdere order operaties uit in één request. Ondersteunt plaatsen, wijzigen en annuleren."
    args_schema: type[BaseModel] = KrakenFuturesBatchOrderInput

    def _run(self, batch_order: str) -> str:
        """Voer batch orders uit op Kraken Futures."""
        result = self._private_request("batchorder", {"batchOrder": batch_order})
        return str(result)


# =============================================================================
# Tool 6: Haal Open Orders Op
# =============================================================================
class KrakenFuturesGetOpenOrdersTool(KrakenFuturesBaseTool):
    """Haal alle open Futures orders op."""

    name: str = "kraken_futures_get_open_orders"
    description: str = "Haal alle momenteel open Futures orders op met details zoals prijs, grootte, type en status."

    def _run(self) -> str:
        """Haal open orders op van Kraken Futures."""
        result = self._private_request("openorders", method="GET")
        return str(result)


# =============================================================================
# Tool 7: Haal Order Status Op
# =============================================================================
class KrakenFuturesGetOrderStatusInput(BaseModel):
    """Input schema voor KrakenFuturesGetOrderStatusTool."""

    order_ids: str | None = Field(
        default=None, description="Komma-gescheiden lijst van order IDs"
    )
    client_order_ids: str | None = Field(
        default=None, description="Komma-gescheiden lijst van client order IDs"
    )


class KrakenFuturesGetOrderStatusTool(KrakenFuturesBaseTool):
    """Haal status op van specifieke orders."""

    name: str = "kraken_futures_get_order_status"
    description: str = "Haal gedetailleerde status op van specifieke orders op basis van order ID of client order ID."
    args_schema: type[BaseModel] = KrakenFuturesGetOrderStatusInput

    def _run(
        self,
        order_ids: str | None = None,
        client_order_ids: str | None = None,
    ) -> str:
        """Haal order status op van Kraken Futures."""
        data: dict[str, Any] = {}
        if order_ids:
            data["orderIds"] = order_ids
        if client_order_ids:
            data["cliOrdIds"] = client_order_ids

        result = self._private_request("orders/status", data, method="GET")
        return str(result)


# =============================================================================
# Tool 8: Dode Man Schakelaar
# =============================================================================
class KrakenFuturesDeadMansSwitchInput(BaseModel):
    """Input schema voor KrakenFuturesDeadMansSwitchTool."""

    timeout: int = Field(
        ...,
        description="Timeout in seconden. Zet op 0 om uit te schakelen. Max 60 seconden.",
    )


class KrakenFuturesDeadMansSwitchTool(KrakenFuturesBaseTool):
    """Stel dode man schakelaar in voor automatische order annulering."""

    name: str = "kraken_futures_dead_mans_switch"
    description: str = "Stel een dode man schakelaar in - annuleer alle orders automatisch indien niet gereset binnen timeout. Veiligheidsmechanisme voor bots."
    args_schema: type[BaseModel] = KrakenFuturesDeadMansSwitchInput

    def _run(self, timeout: int) -> str:
        """Stel dode man schakelaar in op Kraken Futures."""
        result = self._private_request("cancelallordersafter", {"timeout": timeout})
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "KrakenFuturesSendOrderTool",
    "KrakenFuturesEditOrderTool",
    "KrakenFuturesCancelOrderTool",
    "KrakenFuturesCancelAllOrdersTool",
    "KrakenFuturesBatchOrderTool",
    "KrakenFuturesGetOpenOrdersTool",
    "KrakenFuturesGetOrderStatusTool",
    "KrakenFuturesDeadMansSwitchTool",
]
