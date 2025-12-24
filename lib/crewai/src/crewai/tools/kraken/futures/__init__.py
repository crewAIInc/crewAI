"""Kraken Futures API Tools voor CrewAI.

Deze module biedt tools voor interactie met de Kraken Futures cryptocurrency exchange API.
Bevat zowel publieke (marktdata) als private (trading, account) endpoints.

Voorbeeld:
    >>> from crewai.tools.kraken.futures import (
    ...     KrakenFuturesGetTickersTool,
    ...     KrakenFuturesSendOrderTool,
    ... )
    >>>
    >>> # Publieke tool (geen auth nodig)
    >>> tickers = KrakenFuturesGetTickersTool()
    >>> result = tickers.run()
    >>>
    >>> # Private tool (vereist API credentials)
    >>> order_tool = KrakenFuturesSendOrderTool(api_key="...", api_secret="...")
    >>> result = order_tool.run(
    ...     order_type="lmt",
    ...     symbol="PI_XBTUSD",
    ...     side="buy",
    ...     size="1",
    ...     limit_price="50000"
    ... )
"""

from crewai.tools.kraken.futures.base import KrakenFuturesBaseTool

# Market Data (4 tools - PUBLIC)
from crewai.tools.kraken.futures.market_data import (
    KrakenFuturesGetTickersTool,
    KrakenFuturesGetTickerTool,
    KrakenFuturesGetOrderBookTool,
    KrakenFuturesGetTradeHistoryTool,
)

# Order Management (8 tools - PRIVATE)
from crewai.tools.kraken.futures.order_management import (
    KrakenFuturesSendOrderTool,
    KrakenFuturesEditOrderTool,
    KrakenFuturesCancelOrderTool,
    KrakenFuturesCancelAllOrdersTool,
    KrakenFuturesBatchOrderTool,
    KrakenFuturesGetOpenOrdersTool,
    KrakenFuturesGetOrderStatusTool,
    KrakenFuturesDeadMansSwitchTool,
)

# Account (5 tools - PRIVATE)
from crewai.tools.kraken.futures.account import (
    KrakenFuturesGetWalletsTool,
    KrakenFuturesGetOpenPositionsTool,
    KrakenFuturesGetPortfolioMarginTool,
    KrakenFuturesCalculateMarginPnLTool,
    KrakenFuturesGetUnwindQueueTool,
)

# Multi-Collateral (4 tools - PRIVATE)
from crewai.tools.kraken.futures.multi_collateral import (
    KrakenFuturesGetLeverageTool,
    KrakenFuturesSetLeverageTool,
    KrakenFuturesGetPnLCurrencyTool,
    KrakenFuturesSetPnLCurrencyTool,
)

# Transfers (3 tools - PRIVATE)
from crewai.tools.kraken.futures.transfers import (
    KrakenFuturesWalletTransferTool,
    KrakenFuturesSubAccountTransferTool,
    KrakenFuturesWithdrawToSpotTool,
)

# Subaccounts (3 tools - PRIVATE)
from crewai.tools.kraken.futures.subaccounts import (
    KrakenFuturesGetSubaccountsTool,
    KrakenFuturesCheckTradingStatusTool,
    KrakenFuturesUpdateTradingStatusTool,
)

# Instruments (3 tools - PUBLIC)
from crewai.tools.kraken.futures.instruments import (
    KrakenFuturesGetInstrumentsTool,
    KrakenFuturesGetInstrumentStatusTool,
    KrakenFuturesGetInstrumentStatusListTool,
)

# Fee Schedules (2 tools - PRIVATE)
from crewai.tools.kraken.futures.fee_schedules import (
    KrakenFuturesGetFeeSchedulesTool,
    KrakenFuturesGetFeeVolumesTool,
)

# Assignment Program (4 tools - PRIVATE)
from crewai.tools.kraken.futures.assignment import (
    KrakenFuturesListAssignmentProgramsTool,
    KrakenFuturesAddAssignmentPreferenceTool,
    KrakenFuturesDeleteAssignmentPreferenceTool,
    KrakenFuturesListAssignmentHistoryTool,
)

# Historical Data (2 tools)
from crewai.tools.kraken.futures.historical import (
    KrakenFuturesGetFillsTool,
    KrakenFuturesGetFundingRatesTool,
)

# Trading Settings (2 tools - PRIVATE)
from crewai.tools.kraken.futures.trading import (
    KrakenFuturesGetSelfTradeStrategyTool,
    KrakenFuturesSetSelfTradeStrategyTool,
)

# General (1 tool - PRIVATE)
from crewai.tools.kraken.futures.general import (
    KrakenFuturesGetNotificationsTool,
)

__all__ = [
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
