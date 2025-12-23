"""Kraken Spot Account Data Tools - Private endpoints for account information."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.kraken.base import KrakenBaseTool


# =============================================================================
# Tool 1: Get Account Balance
# =============================================================================
class GetAccountBalanceTool(KrakenBaseTool):
    """Get current account balance for all assets."""

    name: str = "kraken_get_account_balance"
    description: str = "Get current account balance for all assets. Returns available balance for each asset."

    def _run(self) -> str:
        """Get account balance from Kraken."""
        result = self._private_request("Balance")
        return str(result)


# =============================================================================
# Tool 2: Get Extended Balance
# =============================================================================
class GetExtendedBalanceTool(KrakenBaseTool):
    """Get extended balance information."""

    name: str = "kraken_get_extended_balance"
    description: str = "Get extended balance info including available balance, hold amounts, and credit for all assets."

    def _run(self) -> str:
        """Get extended balance from Kraken."""
        result = self._private_request("BalanceEx")
        return str(result)


# =============================================================================
# Tool 3: Get Trade Balance
# =============================================================================
class GetTradeBalanceInput(BaseModel):
    """Input schema for GetTradeBalanceTool."""

    asset: str | None = Field(
        default=None, description="Base asset for calculations (default: ZUSD)"
    )


class GetTradeBalanceTool(KrakenBaseTool):
    """Get trade balance including margin information."""

    name: str = "kraken_get_trade_balance"
    description: str = "Get trade balance including equivalent balance, trade balance, margin used, unrealized P&L, and free margin."
    args_schema: type[BaseModel] = GetTradeBalanceInput

    def _run(self, asset: str | None = None) -> str:
        """Get trade balance from Kraken."""
        data: dict[str, Any] = {}
        if asset:
            data["asset"] = asset
        result = self._private_request("TradeBalance", data)
        return str(result)


# =============================================================================
# Tool 4: Get Open Orders
# =============================================================================
class GetOpenOrdersInput(BaseModel):
    """Input schema for GetOpenOrdersTool."""

    trades: bool | None = Field(
        default=None, description="Include trades related to orders in output"
    )
    userref: int | None = Field(
        default=None, description="Filter orders by user reference ID"
    )


class GetOpenOrdersTool(KrakenBaseTool):
    """Get list of all currently open orders."""

    name: str = "kraken_get_open_orders"
    description: str = "Get list of all currently open orders with details like price, volume, type, and status."
    args_schema: type[BaseModel] = GetOpenOrdersInput

    def _run(self, trades: bool | None = None, userref: int | None = None) -> str:
        """Get open orders from Kraken."""
        data: dict[str, Any] = {}
        if trades is not None:
            data["trades"] = trades
        if userref:
            data["userref"] = userref
        result = self._private_request("OpenOrders", data)
        return str(result)


# =============================================================================
# Tool 5: Get Closed Orders
# =============================================================================
class GetClosedOrdersInput(BaseModel):
    """Input schema for GetClosedOrdersTool."""

    trades: bool | None = Field(default=None, description="Include trades in output")
    userref: int | None = Field(
        default=None, description="Filter by user reference ID"
    )
    start: str | None = Field(
        default=None, description="Start timestamp or order txid"
    )
    end: str | None = Field(default=None, description="End timestamp or order txid")
    ofs: int | None = Field(default=None, description="Offset for pagination")
    closetime: str | None = Field(
        default=None, description="Which time to use: 'open', 'close', or 'both'"
    )


class GetClosedOrdersTool(KrakenBaseTool):
    """Get list of closed orders."""

    name: str = "kraken_get_closed_orders"
    description: str = "Get list of closed orders with full details. Supports filtering by time range and pagination."
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
        """Get closed orders from Kraken."""
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
    """Input schema for QueryOrdersInfoTool."""

    txid: str = Field(
        ..., description="Comma-separated list of transaction IDs to query"
    )
    trades: bool | None = Field(default=None, description="Include trades in output")
    userref: int | None = Field(
        default=None, description="Filter by user reference ID"
    )


class QueryOrdersInfoTool(KrakenBaseTool):
    """Query info about specific orders by transaction ID."""

    name: str = "kraken_query_orders_info"
    description: str = "Query detailed info about specific orders by their transaction IDs (up to 50)."
    args_schema: type[BaseModel] = QueryOrdersInput

    def _run(
        self, txid: str, trades: bool | None = None, userref: int | None = None
    ) -> str:
        """Query orders info from Kraken."""
        data: dict[str, Any] = {"txid": txid}
        if trades is not None:
            data["trades"] = trades
        if userref:
            data["userref"] = userref
        result = self._private_request("QueryOrders", data)
        return str(result)


# =============================================================================
# Tool 7: Get Trades History
# =============================================================================
class GetTradesHistoryInput(BaseModel):
    """Input schema for GetTradesHistoryTool."""

    type: str | None = Field(
        default=None,
        description="Trade type filter: 'all', 'any position', 'closed position', 'closing position', 'no position'",
    )
    trades: bool | None = Field(default=None, description="Include related trades")
    start: str | None = Field(default=None, description="Start timestamp")
    end: str | None = Field(default=None, description="End timestamp")
    ofs: int | None = Field(default=None, description="Offset for pagination")


class GetTradesHistoryTool(KrakenBaseTool):
    """Get history of all your trades."""

    name: str = "kraken_get_trades_history"
    description: str = "Get history of all your executed trades with details like price, volume, cost, fee, and margin."
    args_schema: type[BaseModel] = GetTradesHistoryInput

    def _run(
        self,
        type: str | None = None,
        trades: bool | None = None,
        start: str | None = None,
        end: str | None = None,
        ofs: int | None = None,
    ) -> str:
        """Get trades history from Kraken."""
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
    """Input schema for QueryTradesInfoTool."""

    txid: str = Field(..., description="Comma-separated list of trade IDs to query")
    trades: bool | None = Field(default=None, description="Include related trades")


class QueryTradesInfoTool(KrakenBaseTool):
    """Query info about specific trades."""

    name: str = "kraken_query_trades_info"
    description: str = "Query detailed info about specific trades by their transaction IDs (up to 20)."
    args_schema: type[BaseModel] = QueryTradesInput

    def _run(self, txid: str, trades: bool | None = None) -> str:
        """Query trades info from Kraken."""
        data: dict[str, Any] = {"txid": txid}
        if trades is not None:
            data["trades"] = trades
        result = self._private_request("QueryTrades", data)
        return str(result)


# =============================================================================
# Tool 9: Get Open Positions
# =============================================================================
class GetOpenPositionsInput(BaseModel):
    """Input schema for GetOpenPositionsTool."""

    txid: str | None = Field(
        default=None, description="Comma-separated position transaction IDs to filter"
    )
    docalcs: bool | None = Field(
        default=None, description="Include profit/loss calculations"
    )
    consolidation: str | None = Field(
        default=None, description="Consolidation method: 'market' to consolidate by market"
    )


class GetOpenPositionsTool(KrakenBaseTool):
    """Get open margin positions."""

    name: str = "kraken_get_open_positions"
    description: str = "Get open margin positions with details like cost, value, profit/loss, and margin used."
    args_schema: type[BaseModel] = GetOpenPositionsInput

    def _run(
        self,
        txid: str | None = None,
        docalcs: bool | None = None,
        consolidation: str | None = None,
    ) -> str:
        """Get open positions from Kraken."""
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
# Tool 10: Get Ledgers
# =============================================================================
class GetLedgersInput(BaseModel):
    """Input schema for GetLedgersTool."""

    asset: str | None = Field(
        default=None, description="Comma-separated list of assets to filter"
    )
    aclass: str | None = Field(default=None, description="Asset class filter")
    type: str | None = Field(
        default=None,
        description="Ledger type: 'all', 'trade', 'deposit', 'withdrawal', 'transfer', 'margin', 'adjustment', 'rollover', 'credit', 'settled', 'staking', 'dividend', 'sale', 'nft_rebate'",
    )
    start: str | None = Field(default=None, description="Start timestamp")
    end: str | None = Field(default=None, description="End timestamp")
    ofs: int | None = Field(default=None, description="Offset for pagination")


class GetLedgersTool(KrakenBaseTool):
    """Get ledger entries."""

    name: str = "kraken_get_ledgers"
    description: str = "Get ledger entries (deposits, withdrawals, trades, fees, margin, etc.) with full transaction details."
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
        """Get ledgers from Kraken."""
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
    """Input schema for QueryLedgersTool."""

    id: str = Field(..., description="Comma-separated list of ledger IDs to query")
    trades: bool | None = Field(default=None, description="Include related trades")


class QueryLedgersTool(KrakenBaseTool):
    """Query info about specific ledger entries."""

    name: str = "kraken_query_ledgers"
    description: str = "Query detailed info about specific ledger entries by their IDs (up to 20)."
    args_schema: type[BaseModel] = QueryLedgersInput

    def _run(self, id: str, trades: bool | None = None) -> str:
        """Query ledgers from Kraken."""
        data: dict[str, Any] = {"id": id}
        if trades is not None:
            data["trades"] = trades
        result = self._private_request("QueryLedgers", data)
        return str(result)


# =============================================================================
# Tool 12: Get Trade Volume
# =============================================================================
class GetTradeVolumeInput(BaseModel):
    """Input schema for GetTradeVolumeTool."""

    pair: str | None = Field(
        default=None,
        description="Comma-separated list of pairs to get fee info for",
    )


class GetTradeVolumeTool(KrakenBaseTool):
    """Get 30-day trading volume and fee tier."""

    name: str = "kraken_get_trade_volume"
    description: str = "Get 30-day USD trading volume and current fee tier. Can also get fee info for specific pairs."
    args_schema: type[BaseModel] = GetTradeVolumeInput

    def _run(self, pair: str | None = None) -> str:
        """Get trade volume from Kraken."""
        data: dict[str, Any] = {}
        if pair:
            data["pair"] = pair
        result = self._private_request("TradeVolume", data)
        return str(result)


# =============================================================================
# Tool 13: Request Export Report
# =============================================================================
class RequestExportReportInput(BaseModel):
    """Input schema for RequestExportReportTool."""

    report: str = Field(
        ..., description="Report type: 'trades' or 'ledgers'"
    )
    format: str | None = Field(
        default=None, description="Report format: 'CSV' (default) or 'TSV'"
    )
    description: str | None = Field(
        default=None, description="Description for the report"
    )
    starttm: str | None = Field(default=None, description="Start timestamp")
    endtm: str | None = Field(default=None, description="End timestamp")


class RequestExportReportTool(KrakenBaseTool):
    """Request generation of a historical data export."""

    name: str = "kraken_request_export_report"
    description: str = "Request generation of a historical data export (trades or ledgers). Returns a report ID to check status."
    args_schema: type[BaseModel] = RequestExportReportInput

    def _run(
        self,
        report: str,
        format: str | None = None,
        description: str | None = None,
        starttm: str | None = None,
        endtm: str | None = None,
    ) -> str:
        """Request export report from Kraken."""
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
# Tool 14: Get Export Report Status
# =============================================================================
class GetExportReportStatusInput(BaseModel):
    """Input schema for GetExportReportStatusTool."""

    report: str = Field(..., description="Report type: 'trades' or 'ledgers'")


class GetExportReportStatusTool(KrakenBaseTool):
    """Check status of export report requests."""

    name: str = "kraken_get_export_status"
    description: str = "Get status of all export report requests of the specified type."
    args_schema: type[BaseModel] = GetExportReportStatusInput

    def _run(self, report: str) -> str:
        """Get export status from Kraken."""
        result = self._private_request("ExportStatus", {"report": report})
        return str(result)


# =============================================================================
# Tool 15: Retrieve Data Export
# =============================================================================
class RetrieveDataExportInput(BaseModel):
    """Input schema for RetrieveDataExportTool."""

    id: str = Field(..., description="Report ID to retrieve")


class RetrieveDataExportTool(KrakenBaseTool):
    """Download a completed export report."""

    name: str = "kraken_retrieve_export"
    description: str = "Download a completed export report by its ID. Returns the report data."
    args_schema: type[BaseModel] = RetrieveDataExportInput

    def _run(self, id: str) -> str:
        """Retrieve export from Kraken."""
        result = self._private_request("RetrieveExport", {"id": id})
        return str(result)


# =============================================================================
# Tool 16: Delete Export Report
# =============================================================================
class DeleteExportReportInput(BaseModel):
    """Input schema for DeleteExportReportTool."""

    id: str = Field(..., description="Report ID to delete")
    type: str = Field(..., description="Delete type: 'cancel' or 'delete'")


class DeleteExportReportTool(KrakenBaseTool):
    """Delete or cancel an export report."""

    name: str = "kraken_delete_export"
    description: str = "Delete a completed export report or cancel a pending one."
    args_schema: type[BaseModel] = DeleteExportReportInput

    def _run(self, id: str, type: str) -> str:
        """Delete export from Kraken."""
        result = self._private_request("RemoveExport", {"id": id, "type": type})
        return str(result)


# =============================================================================
# Tool 17: Get Order Amends
# =============================================================================
class GetOrderAmendsInput(BaseModel):
    """Input schema for GetOrderAmendsTool."""

    order_txid: str | None = Field(
        default=None, description="Filter by order transaction ID"
    )
    start: str | None = Field(default=None, description="Start timestamp")
    end: str | None = Field(default=None, description="End timestamp")


class GetOrderAmendsTool(KrakenBaseTool):
    """Get order amendment history."""

    name: str = "kraken_get_order_amends"
    description: str = "Get history of order amendments showing what changed on each amend."
    args_schema: type[BaseModel] = GetOrderAmendsInput

    def _run(
        self,
        order_txid: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> str:
        """Get order amends from Kraken."""
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
