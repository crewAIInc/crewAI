"""Kraken API Tools for CrewAI.

This module provides tools for interacting with the Kraken cryptocurrency exchange API.
Includes both public (market data) and private (trading, account) endpoints.

Example:
    >>> from crewai.tools.kraken import GetTickerInformationTool, AddOrderTool
    >>>
    >>> # Public tool (no auth needed)
    >>> ticker = GetTickerInformationTool()
    >>> result = ticker.run(pair="XBTUSD")
    >>>
    >>> # Private tool (requires API credentials)
    >>> order_tool = AddOrderTool(api_key="...", api_secret="...")
    >>> result = order_tool.run(pair="XBTUSD", type="buy", ordertype="limit", volume="0.001", price="50000")
"""

from crewai.tools.kraken.base import KrakenBaseTool
from crewai.tools.kraken.spot import (
    # Market Data (9)
    GetServerTimeTool,
    GetSystemStatusTool,
    GetAssetInfoTool,
    GetTradableAssetPairsTool,
    GetTickerInformationTool,
    GetOrderBookTool,
    GetRecentTradesTool,
    GetRecentSpreadsTool,
    GetOHLCDataTool,
    # Trading (9)
    AddOrderTool,
    AddOrderBatchTool,
    AmendOrderTool,
    EditOrderTool,
    CancelOrderTool,
    CancelOrderBatchTool,
    CancelAllOrdersTool,
    CancelAllOrdersAfterXTool,
    GetWebSocketsTokenTool,
    # Account (17)
    GetAccountBalanceTool,
    GetExtendedBalanceTool,
    GetTradeBalanceTool,
    GetOpenOrdersTool,
    GetClosedOrdersTool,
    QueryOrdersInfoTool,
    GetTradesHistoryTool,
    QueryTradesInfoTool,
    GetOpenPositionsTool,
    GetLedgersTool,
    QueryLedgersTool,
    GetTradeVolumeTool,
    RequestExportReportTool,
    GetExportReportStatusTool,
    RetrieveDataExportTool,
    DeleteExportReportTool,
    GetOrderAmendsTool,
    # Funding (10)
    GetDepositMethodsTool,
    GetDepositAddressesTool,
    GetDepositStatusTool,
    GetWithdrawalMethodsTool,
    GetWithdrawalAddressesTool,
    GetWithdrawalInfoTool,
    WithdrawFundsTool,
    GetWithdrawalStatusTool,
    WalletTransferTool,
    CancelWithdrawalTool,
    # Earn (6)
    ListEarnStrategiesTool,
    ListEarnAllocationsTool,
    AllocateEarnFundsTool,
    GetAllocationStatusTool,
    DeallocateEarnFundsTool,
    GetDeallocationStatusTool,
    # Subaccounts (2)
    CreateSubaccountTool,
    AccountTransferTool,
    # Transparency (2)
    GetPreTradeDataTool,
    GetPostTradeDataTool,
)

__all__ = [
    # Base
    "KrakenBaseTool",
    # Market Data (9) - PUBLIC
    "GetServerTimeTool",
    "GetSystemStatusTool",
    "GetAssetInfoTool",
    "GetTradableAssetPairsTool",
    "GetTickerInformationTool",
    "GetOrderBookTool",
    "GetRecentTradesTool",
    "GetRecentSpreadsTool",
    "GetOHLCDataTool",
    # Trading (9) - PRIVATE
    "AddOrderTool",
    "AddOrderBatchTool",
    "AmendOrderTool",
    "EditOrderTool",
    "CancelOrderTool",
    "CancelOrderBatchTool",
    "CancelAllOrdersTool",
    "CancelAllOrdersAfterXTool",
    "GetWebSocketsTokenTool",
    # Account (17) - PRIVATE
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
    # Funding (10) - PRIVATE
    "GetDepositMethodsTool",
    "GetDepositAddressesTool",
    "GetDepositStatusTool",
    "GetWithdrawalMethodsTool",
    "GetWithdrawalAddressesTool",
    "GetWithdrawalInfoTool",
    "WithdrawFundsTool",
    "GetWithdrawalStatusTool",
    "WalletTransferTool",
    "CancelWithdrawalTool",
    # Earn (6) - PRIVATE
    "ListEarnStrategiesTool",
    "ListEarnAllocationsTool",
    "AllocateEarnFundsTool",
    "GetAllocationStatusTool",
    "DeallocateEarnFundsTool",
    "GetDeallocationStatusTool",
    # Subaccounts (2) - PRIVATE
    "CreateSubaccountTool",
    "AccountTransferTool",
    # Transparency (2) - PUBLIC
    "GetPreTradeDataTool",
    "GetPostTradeDataTool",
]
