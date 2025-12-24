"""Kraken API Tools voor CrewAI.

Deze module biedt tools voor interactie met de Kraken cryptocurrency exchange APIs.
Bevat zowel Spot als Futures tools met publieke en private endpoints.

Voorbeeld:
    >>> from crewai.tools.kraken import (
    ...     # Spot tools
    ...     GetTickerInformationTool,
    ...     AddOrderTool,
    ...     # Futures tools
    ...     KrakenFuturesGetTickersTool,
    ...     KrakenFuturesSendOrderTool,
    ... )
    >>>
    >>> # Spot publieke tool (geen auth nodig)
    >>> ticker = GetTickerInformationTool()
    >>> result = ticker.run(pair="XBTUSD")
    >>>
    >>> # Futures private tool (vereist API credentials)
    >>> order_tool = KrakenFuturesSendOrderTool(api_key="...", api_secret="...")
    >>> result = order_tool.run(
    ...     order_type="lmt",
    ...     symbol="PI_XBTUSD",
    ...     side="buy",
    ...     size="1",
    ...     limit_price="50000"
    ... )
"""

# ============================================================================
# SPOT API Tools (55 tools)
# ============================================================================
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

# ============================================================================
# FUTURES API Tools (41 tools)
# ============================================================================
from crewai.tools.kraken.futures import (
    # Base
    KrakenFuturesBaseTool,
    # Market Data (4) - PUBLIC
    KrakenFuturesGetTickersTool,
    KrakenFuturesGetTickerTool,
    KrakenFuturesGetOrderBookTool,
    KrakenFuturesGetTradeHistoryTool,
    # Order Management (8) - PRIVATE
    KrakenFuturesSendOrderTool,
    KrakenFuturesEditOrderTool,
    KrakenFuturesCancelOrderTool,
    KrakenFuturesCancelAllOrdersTool,
    KrakenFuturesBatchOrderTool,
    KrakenFuturesGetOpenOrdersTool,
    KrakenFuturesGetOrderStatusTool,
    KrakenFuturesDeadMansSwitchTool,
    # Account (5) - PRIVATE
    KrakenFuturesGetWalletsTool,
    KrakenFuturesGetOpenPositionsTool,
    KrakenFuturesGetPortfolioMarginTool,
    KrakenFuturesCalculateMarginPnLTool,
    KrakenFuturesGetUnwindQueueTool,
    # Multi-Collateral (4) - PRIVATE
    KrakenFuturesGetLeverageTool,
    KrakenFuturesSetLeverageTool,
    KrakenFuturesGetPnLCurrencyTool,
    KrakenFuturesSetPnLCurrencyTool,
    # Transfers (3) - PRIVATE
    KrakenFuturesWalletTransferTool,
    KrakenFuturesSubAccountTransferTool,
    KrakenFuturesWithdrawToSpotTool,
    # Subaccounts (3) - PRIVATE
    KrakenFuturesGetSubaccountsTool,
    KrakenFuturesCheckTradingStatusTool,
    KrakenFuturesUpdateTradingStatusTool,
    # Instruments (3) - PUBLIC
    KrakenFuturesGetInstrumentsTool,
    KrakenFuturesGetInstrumentStatusTool,
    KrakenFuturesGetInstrumentStatusListTool,
    # Fee Schedules (2) - PRIVATE
    KrakenFuturesGetFeeSchedulesTool,
    KrakenFuturesGetFeeVolumesTool,
    # Assignment Program (4) - PRIVATE
    KrakenFuturesListAssignmentProgramsTool,
    KrakenFuturesAddAssignmentPreferenceTool,
    KrakenFuturesDeleteAssignmentPreferenceTool,
    KrakenFuturesListAssignmentHistoryTool,
    # Historical Data (2)
    KrakenFuturesGetFillsTool,
    KrakenFuturesGetFundingRatesTool,
    # Trading Settings (2) - PRIVATE
    KrakenFuturesGetSelfTradeStrategyTool,
    KrakenFuturesSetSelfTradeStrategyTool,
    # General (1) - PRIVATE
    KrakenFuturesGetNotificationsTool,
)

__all__ = [
    # ========================================================================
    # SPOT API (55 tools)
    # ========================================================================
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
    # ========================================================================
    # FUTURES API (41 tools)
    # ========================================================================
    # Base
    "KrakenFuturesBaseTool",
    # Market Data (4) - PUBLIC
    "KrakenFuturesGetTickersTool",
    "KrakenFuturesGetTickerTool",
    "KrakenFuturesGetOrderBookTool",
    "KrakenFuturesGetTradeHistoryTool",
    # Order Management (8) - PRIVATE
    "KrakenFuturesSendOrderTool",
    "KrakenFuturesEditOrderTool",
    "KrakenFuturesCancelOrderTool",
    "KrakenFuturesCancelAllOrdersTool",
    "KrakenFuturesBatchOrderTool",
    "KrakenFuturesGetOpenOrdersTool",
    "KrakenFuturesGetOrderStatusTool",
    "KrakenFuturesDeadMansSwitchTool",
    # Account (5) - PRIVATE
    "KrakenFuturesGetWalletsTool",
    "KrakenFuturesGetOpenPositionsTool",
    "KrakenFuturesGetPortfolioMarginTool",
    "KrakenFuturesCalculateMarginPnLTool",
    "KrakenFuturesGetUnwindQueueTool",
    # Multi-Collateral (4) - PRIVATE
    "KrakenFuturesGetLeverageTool",
    "KrakenFuturesSetLeverageTool",
    "KrakenFuturesGetPnLCurrencyTool",
    "KrakenFuturesSetPnLCurrencyTool",
    # Transfers (3) - PRIVATE
    "KrakenFuturesWalletTransferTool",
    "KrakenFuturesSubAccountTransferTool",
    "KrakenFuturesWithdrawToSpotTool",
    # Subaccounts (3) - PRIVATE
    "KrakenFuturesGetSubaccountsTool",
    "KrakenFuturesCheckTradingStatusTool",
    "KrakenFuturesUpdateTradingStatusTool",
    # Instruments (3) - PUBLIC
    "KrakenFuturesGetInstrumentsTool",
    "KrakenFuturesGetInstrumentStatusTool",
    "KrakenFuturesGetInstrumentStatusListTool",
    # Fee Schedules (2) - PRIVATE
    "KrakenFuturesGetFeeSchedulesTool",
    "KrakenFuturesGetFeeVolumesTool",
    # Assignment Program (4) - PRIVATE
    "KrakenFuturesListAssignmentProgramsTool",
    "KrakenFuturesAddAssignmentPreferenceTool",
    "KrakenFuturesDeleteAssignmentPreferenceTool",
    "KrakenFuturesListAssignmentHistoryTool",
    # Historical Data (2)
    "KrakenFuturesGetFillsTool",
    "KrakenFuturesGetFundingRatesTool",
    # Trading Settings (2) - PRIVATE
    "KrakenFuturesGetSelfTradeStrategyTool",
    "KrakenFuturesSetSelfTradeStrategyTool",
    # General (1) - PRIVATE
    "KrakenFuturesGetNotificationsTool",
]
