"""Kraken Spot Account Data Tools - Private endpoints voor account informatie."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


def _balance_threshold() -> float:
    raw = os.getenv("KRAKEN_BALANCE_MIN_THRESHOLD", "0.000001").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.000001
    return max(value, 0.0)


def _has_significant_value(value: Any, threshold: float) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_significant_value(v, threshold) for v in value.values())
    if isinstance(value, list):
        return any(_has_significant_value(v, threshold) for v in value)
    if isinstance(value, bool):
        return bool(value)
    try:
        return abs(float(value)) >= threshold
    except (TypeError, ValueError):
        return True


def _filter_balance_result(payload: dict[str, Any], threshold: float) -> dict[str, Any]:
    result = payload.get("result")
    if not isinstance(result, dict):
        return payload
    filtered = {
        asset: data
        for asset, data in result.items()
        if _has_significant_value(data, threshold)
    }
    if filtered == result:
        return payload
    new_payload = dict(payload)
    new_payload["result"] = filtered
    return new_payload


# =============================================================================
# Tool 1: Haal Account Saldo Op
# =============================================================================
class GetAccountBalanceTool(KrakenBaseTool):
    """Haal huidig account saldo op voor alle assets."""

    name: str = "kraken_get_account_balance"
    description: str = "Haal huidig account saldo op voor alle assets. Geeft beschikbaar saldo terug voor elk asset."

    def _run(self) -> str:
        """Haal account saldo op van Kraken."""
        result = self._private_request("Balance")
        return str(_filter_balance_result(result, _balance_threshold()))


# =============================================================================
# Tool 2: Haal Uitgebreid Saldo Op
# =============================================================================
class GetExtendedBalanceTool(KrakenBaseTool):
    """Haal uitgebreide saldo informatie op."""

    name: str = "kraken_get_extended_balance"
    description: str = "Haal uitgebreide saldo info op inclusief beschikbaar saldo, vastgehouden bedragen en krediet voor alle assets."

    def _run(self) -> str:
        """Haal uitgebreid saldo op van Kraken."""
        result = self._private_request("BalanceEx")
        return str(_filter_balance_result(result, _balance_threshold()))


# =============================================================================
# Tool 3: Haal Handelsaldo Op
# =============================================================================
class GetTradeBalanceInput(BaseModel):
    """Input schema voor GetTradeBalanceTool."""

    asset: str | None = Field(
        default=None, description="Basis asset voor berekeningen (standaard: ZUSD)"
    )


class GetTradeBalanceTool(KrakenBaseTool):
    """Haal handelsaldo op inclusief margin informatie."""

    name: str = "kraken_get_trade_balance"
    description: str = "Haal handelsaldo op inclusief equivalent saldo, handelsaldo, gebruikte margin, ongerealiseerde W&V en vrije margin."
    args_schema: type[BaseModel] = GetTradeBalanceInput

    def _run(self, asset: str | None = None) -> str:
        """Haal handelsaldo op van Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        result = self._private_request("TradeBalance", data)
        return str(result)


# =============================================================================
# Tool 4: Haal Open Orders Op
# =============================================================================
class GetOpenOrdersInput(BaseModel):
    """Input schema voor GetOpenOrdersTool."""

    trades: bool | None = Field(
        default=None, description="Voeg trades gerelateerd aan orders toe in output"
    )
    userref: int | None = Field(
        default=None, description="Filter orders op gebruiker referentie ID"
    )


class GetOpenOrdersTool(KrakenBaseTool):
    """Haal lijst op van alle momenteel open orders."""

    name: str = "kraken_get_open_orders"
    description: str = "Haal lijst op van alle momenteel open orders met details zoals prijs, volume, type en status."
    args_schema: type[BaseModel] = GetOpenOrdersInput

    def _run(self, trades: bool | None = None, userref: int | None = None) -> str:
        """Haal open orders op van Kraken."""
        data: dict[str, Any] = {}
        if trades is not None:
            data["trades"] = trades
        if userref:
            data["userref"] = userref
        result = self._private_request("OpenOrders", data)
        return str(result)


# =============================================================================
# Tool 5: Haal Gesloten Orders Op
# =============================================================================
class GetClosedOrdersInput(BaseModel):
    """Input schema voor GetClosedOrdersTool."""

    trades: bool | None = Field(default=None, description="Voeg trades toe in output")
    userref: int | None = Field(
        default=None, description="Filter op gebruiker referentie ID"
    )
    start: str | None = Field(
        default=None, description="Start timestamp of order txid"
    )
    end: str | None = Field(default=None, description="Eind timestamp of order txid")
    ofs: int | None = Field(default=None, description="Offset voor paginatie")
    closetime: str | None = Field(
        default=None, description="Welke tijd te gebruiken: 'open', 'close' of 'both'"
    )


class GetClosedOrdersTool(KrakenBaseTool):
    """Haal lijst op van gesloten orders."""

    name: str = "kraken_get_closed_orders"
    description: str = "Haal lijst op van gesloten orders met volledige details. Ondersteunt filteren op tijdsbereik en paginatie."
    args_schema: type[BaseModel] = GetClosedOrdersInput

    def _run(
        self,
        trades: bool | None = None,
        userref: int | None = None,
        start: str | None = None,
        end: str | None = None,
        ofs: int | None = None,
        closetime: str | None = None,
    ) -> str:
        """Haal gesloten orders op van Kraken."""
        data: dict[str, Any] = {}
        if trades is not None:
            data["trades"] = trades
        if userref:
            data["userref"] = userref
        if start:
            data["start"] = start
        if end:
            data["end"] = end
        if ofs:
            data["ofs"] = ofs
        if closetime:
            data["closetime"] = closetime
        result = self._private_request("ClosedOrders", data)
        return str(result)


# =============================================================================
# Tool 6: Query Orders Info
# =============================================================================
class QueryOrdersInput(BaseModel):
    """Input schema voor QueryOrdersInfoTool."""

    txid: str = Field(
        ..., description="Komma-gescheiden lijst van transactie IDs om te queryen"
    )
    trades: bool | None = Field(default=None, description="Voeg trades toe in output")
    userref: int | None = Field(
        default=None, description="Filter op gebruiker referentie ID"
    )


class QueryOrdersInfoTool(KrakenBaseTool):
    """Query info over specifieke orders op transactie ID."""

    name: str = "kraken_query_orders_info"
    description: str = "Query gedetailleerde info over specifieke orders op hun transactie IDs (tot 50)."
    args_schema: type[BaseModel] = QueryOrdersInput

    def _run(
        self, txid: str, trades: bool | None = None, userref: int | None = None
    ) -> str:
        """Query orders info van Kraken."""
        data: dict[str, Any] = {"txid": txid}
        if trades is not None:
            data["trades"] = trades
        if userref:
            data["userref"] = userref
        result = self._private_request("QueryOrders", data)
        return str(result)


# =============================================================================
# Tool 7: Haal Trades Geschiedenis Op
# =============================================================================
class GetTradesHistoryInput(BaseModel):
    """Input schema voor GetTradesHistoryTool."""

    type: str | None = Field(
        default=None,
        description="Trade type filter: 'all', 'any position', 'closed position', 'closing position', 'no position'",
    )
    trades: bool | None = Field(default=None, description="Voeg gerelateerde trades toe")
    start: str | None = Field(default=None, description="Start timestamp")
    end: str | None = Field(default=None, description="Eind timestamp")
    ofs: int | None = Field(default=None, description="Offset voor paginatie")


class GetTradesHistoryTool(KrakenBaseTool):
    """Haal geschiedenis op van al je trades."""

    name: str = "kraken_get_trades_history"
    description: str = "Haal geschiedenis op van al je uitgevoerde trades met details zoals prijs, volume, kosten, fee en margin."
    args_schema: type[BaseModel] = GetTradesHistoryInput

    def _run(
        self,
        type: str | None = None,
        trades: bool | None = None,
        start: str | None = None,
        end: str | None = None,
        ofs: int | None = None,
    ) -> str:
        """Haal trades geschiedenis op van Kraken."""
        data: dict[str, Any] = {}
        if type:
            data["type"] = type
        if trades is not None:
            data["trades"] = trades
        if start:
            data["start"] = start
        if end:
            data["end"] = end
        if ofs:
            data["ofs"] = ofs
        result = self._private_request("TradesHistory", data)
        return str(result)


# =============================================================================
# Tool 8: Query Trades Info
# =============================================================================
class QueryTradesInput(BaseModel):
    """Input schema voor QueryTradesInfoTool."""

    txid: str = Field(..., description="Komma-gescheiden lijst van trade IDs om te queryen")
    trades: bool | None = Field(default=None, description="Voeg gerelateerde trades toe")


class QueryTradesInfoTool(KrakenBaseTool):
    """Query info over specifieke trades."""

    name: str = "kraken_query_trades_info"
    description: str = "Query gedetailleerde info over specifieke trades op hun transactie IDs (tot 20)."
    args_schema: type[BaseModel] = QueryTradesInput

    def _run(self, txid: str, trades: bool | None = None) -> str:
        """Query trades info van Kraken."""
        data: dict[str, Any] = {"txid": txid}
        if trades is not None:
            data["trades"] = trades
        result = self._private_request("QueryTrades", data)
        return str(result)


# =============================================================================
# Tool 9: Haal Open Posities Op
# =============================================================================
class GetOpenPositionsInput(BaseModel):
    """Input schema voor GetOpenPositionsTool."""

    txid: str | None = Field(
        default=None, description="Komma-gescheiden positie transactie IDs om te filteren"
    )
    docalcs: bool | None = Field(
        default=None, description="Voeg winst/verlies berekeningen toe"
    )
    consolidation: str | None = Field(
        default=None, description="Consolidatie methode: 'market' om per markt te consolideren"
    )


class GetOpenPositionsTool(KrakenBaseTool):
    """Haal open margin posities op."""

    name: str = "kraken_get_open_positions"
    description: str = "Haal open margin posities op met details zoals kosten, waarde, winst/verlies en gebruikte margin."
    args_schema: type[BaseModel] = GetOpenPositionsInput

    def _run(
        self,
        txid: str | None = None,
        docalcs: bool | None = None,
        consolidation: str | None = None,
    ) -> str:
        """Haal open posities op van Kraken."""
        data: dict[str, Any] = {}
        if txid:
            data["txid"] = txid
        if docalcs is not None:
            data["docalcs"] = docalcs
        if consolidation:
            data["consolidation"] = consolidation
        result = self._private_request("OpenPositions", data)
        return str(result)


# =============================================================================
# Tool 10: Haal Ledgers Op
# =============================================================================
class GetLedgersInput(BaseModel):
    """Input schema voor GetLedgersTool."""

    asset: str | None = Field(
        default=None, description="Komma-gescheiden lijst van assets om te filteren"
    )
    aclass: str | None = Field(default=None, description="Asset klasse filter")
    type: str | None = Field(
        default=None,
        description="Ledger type: 'all', 'trade', 'deposit', 'withdrawal', 'transfer', 'margin', 'adjustment', 'rollover', 'credit', 'settled', 'staking', 'dividend', 'sale', 'nft_rebate'",
    )
    start: str | None = Field(default=None, description="Start timestamp")
    end: str | None = Field(default=None, description="Eind timestamp")
    ofs: int | None = Field(default=None, description="Offset voor paginatie")


class GetLedgersTool(KrakenBaseTool):
    """Haal ledger entries op."""

    name: str = "kraken_get_ledgers"
    description: str = "Haal ledger entries op (stortingen, opnames, trades, fees, margin, etc.) met volledige transactie details."
    args_schema: type[BaseModel] = GetLedgersInput

    def _run(
        self,
        asset: str | None = None,
        aclass: str | None = None,
        type: str | None = None,
        start: str | None = None,
        end: str | None = None,
        ofs: int | None = None,
    ) -> str:
        """Haal ledgers op van Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        if aclass:
            data["aclass"] = aclass
        if type:
            data["type"] = type
        if start:
            data["start"] = start
        if end:
            data["end"] = end
        if ofs:
            data["ofs"] = ofs
        result = self._private_request("Ledgers", data)
        return str(result)


# =============================================================================
# Tool 11: Query Ledgers
# =============================================================================
class QueryLedgersInput(BaseModel):
    """Input schema voor QueryLedgersTool."""

    id: str = Field(..., description="Komma-gescheiden lijst van ledger IDs om te queryen")
    trades: bool | None = Field(default=None, description="Voeg gerelateerde trades toe")


class QueryLedgersTool(KrakenBaseTool):
    """Query info over specifieke ledger entries."""

    name: str = "kraken_query_ledgers"
    description: str = "Query gedetailleerde info over specifieke ledger entries op hun IDs (tot 20)."
    args_schema: type[BaseModel] = QueryLedgersInput

    def _run(self, id: str, trades: bool | None = None) -> str:
        """Query ledgers van Kraken."""
        data: dict[str, Any] = {"id": id}
        if trades is not None:
            data["trades"] = trades
        result = self._private_request("QueryLedgers", data)
        return str(result)


# =============================================================================
# Tool 12: Haal Handelsvolume Op
# =============================================================================
class GetTradeVolumeInput(BaseModel):
    """Input schema voor GetTradeVolumeTool."""

    pair: str | None = Field(
        default=None,
        description="Komma-gescheiden lijst van paren om fee info voor op te halen",
    )


class GetTradeVolumeTool(KrakenBaseTool):
    """Haal 30-dagen handelsvolume en fee tier op."""

    name: str = "kraken_get_trade_volume"
    description: str = "Haal 30-dagen USD handelsvolume en huidige fee tier op. Kan ook fee info ophalen voor specifieke paren."
    args_schema: type[BaseModel] = GetTradeVolumeInput

    def _run(self, pair: str | None = None) -> str:
        """Haal handelsvolume op van Kraken."""
        data: dict[str, Any] = {}
        if pair:
            data["pair"] = pair
        result = self._private_request("TradeVolume", data)
        return str(result)


# =============================================================================
# Tool 13: Vraag Export Rapport Aan
# =============================================================================
class RequestExportReportInput(BaseModel):
    """Input schema voor RequestExportReportTool."""

    report: str = Field(
        ..., description="Rapport type: 'trades' of 'ledgers'"
    )
    format: str | None = Field(
        default=None, description="Rapport formaat: 'CSV' (standaard) of 'TSV'"
    )
    description: str | None = Field(
        default=None, description="Beschrijving voor het rapport"
    )
    starttm: str | None = Field(default=None, description="Start timestamp")
    endtm: str | None = Field(default=None, description="Eind timestamp")


class RequestExportReportTool(KrakenBaseTool):
    """Vraag generatie van een historische data export aan."""

    name: str = "kraken_request_export_report"
    description: str = "Vraag generatie van een historische data export aan (trades of ledgers). Geeft een rapport ID terug om status te checken."
    args_schema: type[BaseModel] = RequestExportReportInput

    def _run(
        self,
        report: str,
        format: str | None = None,
        description: str | None = None,
        starttm: str | None = None,
        endtm: str | None = None,
    ) -> str:
        """Vraag export rapport aan van Kraken."""
        data: dict[str, Any] = {"report": report}
        if format:
            data["format"] = format
        if description:
            data["description"] = description
        if starttm:
            data["starttm"] = starttm
        if endtm:
            data["endtm"] = endtm
        result = self._private_request("AddExport", data)
        return str(result)


# =============================================================================
# Tool 14: Haal Export Rapport Status Op
# =============================================================================
class GetExportReportStatusInput(BaseModel):
    """Input schema voor GetExportReportStatusTool."""

    report: str = Field(..., description="Rapport type: 'trades' of 'ledgers'")


class GetExportReportStatusTool(KrakenBaseTool):
    """Check status van export rapport aanvragen."""

    name: str = "kraken_get_export_status"
    description: str = "Haal status op van alle export rapport aanvragen van het gespecificeerde type."
    args_schema: type[BaseModel] = GetExportReportStatusInput

    def _run(self, report: str) -> str:
        """Haal export status op van Kraken."""
        result = self._private_request("ExportStatus", {"report": report})
        return str(result)


# =============================================================================
# Tool 15: Haal Data Export Op
# =============================================================================
class RetrieveDataExportInput(BaseModel):
    """Input schema voor RetrieveDataExportTool."""

    id: str = Field(..., description="Rapport ID om op te halen")


class RetrieveDataExportTool(KrakenBaseTool):
    """Download een voltooid export rapport."""

    name: str = "kraken_retrieve_export"
    description: str = "Download een voltooid export rapport op basis van ID. Geeft de rapport data terug."
    args_schema: type[BaseModel] = RetrieveDataExportInput

    def _run(self, id: str) -> str:
        """Haal export op van Kraken."""
        result = self._private_request("RetrieveExport", {"id": id})
        return str(result)


# =============================================================================
# Tool 16: Verwijder Export Rapport
# =============================================================================
class DeleteExportReportInput(BaseModel):
    """Input schema voor DeleteExportReportTool."""

    id: str = Field(..., description="Rapport ID om te verwijderen")
    type: str = Field(..., description="Verwijder type: 'cancel' of 'delete'")


class DeleteExportReportTool(KrakenBaseTool):
    """Verwijder of annuleer een export rapport."""

    name: str = "kraken_delete_export"
    description: str = "Verwijder een voltooid export rapport of annuleer een in afwachting zijnde."
    args_schema: type[BaseModel] = DeleteExportReportInput

    def _run(self, id: str, type: str) -> str:
        """Verwijder export van Kraken."""
        result = self._private_request("RemoveExport", {"id": id, "type": type})
        return str(result)


# =============================================================================
# Tool 17: Haal Order Wijzigingen Op
# =============================================================================
class GetOrderAmendsInput(BaseModel):
    """Input schema voor GetOrderAmendsTool."""

    order_txid: str | None = Field(
        default=None, description="Filter op order transactie ID"
    )
    start: str | None = Field(default=None, description="Start timestamp")
    end: str | None = Field(default=None, description="Eind timestamp")


class GetOrderAmendsTool(KrakenBaseTool):
    """Haal order wijzigingsgeschiedenis op."""

    name: str = "kraken_get_order_amends"
    description: str = "Haal geschiedenis op van order wijzigingen die laten zien wat er bij elke wijziging veranderd is."
    args_schema: type[BaseModel] = GetOrderAmendsInput

    def _run(
        self,
        order_txid: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> str:
        """Haal order wijzigingen op van Kraken."""
        data: dict[str, Any] = {}
        if order_txid:
            data["order_txid"] = order_txid
        if start:
            data["start"] = start
        if end:
            data["end"] = end
        result = self._private_request("OrderAmends", data)
        return str(result)


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "GetAccountBalanceTool",
    "GetExtendedBalanceTool",
    "GetTradeBalanceTool",
    "GetOpenOrdersTool",
    "GetClosedOrdersTool",
    "QueryOrdersInfoTool",
    "GetTradesHistoryTool",
    "QueryTradesInfoTool",
    "GetOpenPositionsTool",
    "GetLedgersTool",
    "QueryLedgersTool",
    "GetTradeVolumeTool",
    "RequestExportReportTool",
    "GetExportReportStatusTool",
    "RetrieveDataExportTool",
    "DeleteExportReportTool",
    "GetOrderAmendsTool",
]
