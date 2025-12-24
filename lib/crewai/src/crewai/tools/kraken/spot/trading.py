"""Kraken Spot Trading Tools - Private endpoints voor orderbeheer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Plaats Order
# =============================================================================
class AddOrderInput(BaseModel):
    """Input schema voor AddOrderTool."""

    pair: str = Field(..., description="Asset paar (bijv. 'XBTUSD')")
    type: str = Field(..., description="Order richting: 'buy' of 'sell'")
    ordertype: str = Field(
        ...,
        description="Order type: 'market', 'limit', 'stop-loss', 'take-profit', 'stop-loss-limit', 'take-profit-limit', 'trailing-stop', 'trailing-stop-limit', 'settle-position'",
    )
    volume: str = Field(..., description="Order volume in basisvaluta")
    price: str | None = Field(
        default=None, description="Prijs (vereist voor limit orders)"
    )
    price2: str | None = Field(
        default=None, description="Secundaire prijs (voor stop-loss-limit, etc.)"
    )
    leverage: str | None = Field(
        default=None, description="Hefboom hoeveelheid (bijv. '2:1', '3:1')"
    )
    reduce_only: bool | None = Field(
        default=None, description="Indien true, zal order alleen bestaande positie verkleinen"
    )
    validate: bool | None = Field(
        default=None, description="Indien true, valideer alleen inputs zonder order te plaatsen"
    )
    userref: int | None = Field(
        default=None, description="Gebruiker referentie ID voor de order"
    )
    oflags: str | None = Field(
        default=None,
        description="Order vlaggen: 'post' (post-only), 'fcib' (fee in basis), 'fciq' (fee in quote), 'nompp' (geen marktprijs bescherming)",
    )


class AddOrderTool(KrakenBaseTool):
    """Plaats een nieuwe spot order op Kraken."""

    name: str = "kraken_add_order"
    description: str = "Plaats een nieuwe spot order op Kraken. Ondersteunt market, limit, stop-loss, take-profit en andere order types."
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
        """Plaats een nieuwe order op Kraken."""
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
# Tool 2: Plaats Order Batch
# =============================================================================
class AddOrderBatchInput(BaseModel):
    """Input schema voor AddOrderBatchTool."""

    pair: str = Field(..., description="Asset paar voor alle orders (bijv. 'XBTUSD')")
    orders: str = Field(
        ...,
        description="JSON array van order objecten, elk met: type, ordertype, volume, price, etc. Max 15 orders.",
    )
    validate: bool | None = Field(
        default=None, description="Indien true, valideer alleen inputs"
    )


class AddOrderBatchTool(KrakenBaseTool):
    """Plaats meerdere orders in één request."""

    name: str = "kraken_add_order_batch"
    description: str = "Plaats meerdere orders (tot 15) in één request voor hetzelfde asset paar."
    args_schema: type[BaseModel] = AddOrderBatchInput

    def _run(
        self, pair: str, orders: str, validate: bool | None = None
    ) -> str:
        """Plaats meerdere orders op Kraken."""
        data: dict[str, Any] = {"pair": pair, "orders": orders}
        if validate is not None:
            data["validate"] = validate
        result = self._private_request("AddOrderBatch", data)
        return str(result)


# =============================================================================
# Tool 3: Wijzig Order
# =============================================================================
class AmendOrderInput(BaseModel):
    """Input schema voor AmendOrderTool."""

    txid: str = Field(..., description="Transactie ID van de order om te wijzigen")
    pair: str = Field(..., description="Asset paar (bijv. 'XBTUSD')")
    volume: str | None = Field(default=None, description="Nieuw order volume")
    price: str | None = Field(default=None, description="Nieuwe order prijs")
    price2: str | None = Field(default=None, description="Nieuwe secundaire prijs")


class AmendOrderTool(KrakenBaseTool):
    """Wijzig een bestaande open order."""

    name: str = "kraken_amend_order"
    description: str = "Wijzig een bestaande open order door volume of prijs aan te passen. Sneller dan edit omdat het niet annuleert/hercreëert."
    args_schema: type[BaseModel] = AmendOrderInput

    def _run(
        self,
        txid: str,
        pair: str,
        volume: str | None = None,
        price: str | None = None,
        price2: str | None = None,
    ) -> str:
        """Wijzig een order op Kraken."""
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
# Tool 4: Bewerk Order
# =============================================================================
class EditOrderInput(BaseModel):
    """Input schema voor EditOrderTool."""

    txid: str = Field(..., description="Transactie ID van de order om te bewerken")
    pair: str = Field(..., description="Asset paar (bijv. 'XBTUSD')")
    volume: str | None = Field(default=None, description="Nieuw order volume")
    price: str | None = Field(default=None, description="Nieuwe order prijs")
    price2: str | None = Field(default=None, description="Nieuwe secundaire prijs")
    oflags: str | None = Field(default=None, description="Nieuwe order vlaggen")
    validate: bool | None = Field(
        default=None, description="Indien true, valideer alleen inputs"
    )


class EditOrderTool(KrakenBaseTool):
    """Bewerk een bestaande open order (annuleer en vervang)."""

    name: str = "kraken_edit_order"
    description: str = "Bewerk een bestaande open order. Dit annuleert de originele order en maakt een nieuwe met bijgewerkte parameters."
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
        """Bewerk een order op Kraken."""
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
# Tool 5: Annuleer Order
# =============================================================================
class CancelOrderInput(BaseModel):
    """Input schema voor CancelOrderTool."""

    txid: str = Field(
        ...,
        description="Transactie ID van order om te annuleren (of komma-gescheiden lijst, of userref met # prefix)",
    )


class CancelOrderTool(KrakenBaseTool):
    """Annuleer een specifieke open order."""

    name: str = "kraken_cancel_order"
    description: str = "Annuleer een specifieke open order op basis van transactie ID. Kan ook userref gebruiken met # prefix."
    args_schema: type[BaseModel] = CancelOrderInput

    def _run(self, txid: str) -> str:
        """Annuleer een order op Kraken."""
        result = self._private_request("CancelOrder", {"txid": txid})
        return str(result)


# =============================================================================
# Tool 6: Annuleer Order Batch
# =============================================================================
class CancelOrderBatchInput(BaseModel):
    """Input schema voor CancelOrderBatchTool."""

    orders: str = Field(
        ...,
        description="JSON array van transactie IDs om te annuleren (bijv. '[\"TXID1\", \"TXID2\"]')",
    )


class CancelOrderBatchTool(KrakenBaseTool):
    """Annuleer meerdere orders in één request."""

    name: str = "kraken_cancel_order_batch"
    description: str = "Annuleer meerdere orders in één request met hun transactie IDs."
    args_schema: type[BaseModel] = CancelOrderBatchInput

    def _run(self, orders: str) -> str:
        """Annuleer meerdere orders op Kraken."""
        result = self._private_request("CancelOrderBatch", {"orders": orders})
        return str(result)


# =============================================================================
# Tool 7: Annuleer Alle Orders
# =============================================================================
class CancelAllOrdersTool(KrakenBaseTool):
    """Annuleer alle open orders."""

    name: str = "kraken_cancel_all_orders"
    description: str = "Annuleer ALLE open orders. Gebruik met voorzichtigheid! Geeft aantal geannuleerde orders terug."

    def _run(self) -> str:
        """Annuleer alle orders op Kraken."""
        result = self._private_request("CancelAll")
        return str(result)


# =============================================================================
# Tool 8: Annuleer Alle Orders Na X
# =============================================================================
class CancelAllOrdersAfterXInput(BaseModel):
    """Input schema voor CancelAllOrdersAfterXTool."""

    timeout: int = Field(
        ...,
        description="Timeout in seconden waarna alle orders geannuleerd worden. Zet op 0 om uit te schakelen.",
    )


class CancelAllOrdersAfterXTool(KrakenBaseTool):
    """Stel een dode man schakelaar in om alle orders te annuleren na timeout."""

    name: str = "kraken_cancel_all_orders_after_x"
    description: str = "Stel een dode man schakelaar in - annuleer alle orders na X seconden indien niet gereset. Handig voor veiligheid. Zet timeout op 0 om uit te schakelen."
    args_schema: type[BaseModel] = CancelAllOrdersAfterXInput

    def _run(self, timeout: int) -> str:
        """Stel dode man schakelaar in op Kraken."""
        result = self._private_request("CancelAllOrdersAfter", {"timeout": timeout})
        return str(result)


# =============================================================================
# Tool 9: Haal WebSockets Token Op
# =============================================================================
class GetWebSocketsTokenTool(KrakenBaseTool):
    """Haal authenticatie token op voor private WebSocket feeds."""

    name: str = "kraken_get_websockets_token"
    description: str = "Haal een authenticatie token op voor verbinding met Kraken's private WebSocket feeds."

    def _run(self) -> str:
        """Haal WebSocket token op van Kraken."""
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
