"""Kraken Futures API tools for QRI Trading Organization.

Provides tool factory functions that return configured tool instances
with API credentials from settings.
"""

from functools import lru_cache

from crewai.tools.kraken import (
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

from krakenagents.config import get_settings


def _get_credentials() -> dict[str, str]:
    """Get Futures API credentials from settings."""
    settings = get_settings()
    return {
        "api_key": settings.kraken_futures_api_key,
        "api_secret": settings.kraken_futures_api_secret,
    }


# =============================================================================
# Category-based tool getters (for specific agent types)
# =============================================================================


@lru_cache
def get_futures_market_tools() -> list:
    """Get public market data tools (no auth required)."""
    return [
        KrakenFuturesGetTickersTool(),
        KrakenFuturesGetTickerTool(),
        KrakenFuturesGetOrderBookTool(),
        KrakenFuturesGetTradeHistoryTool(),
        KrakenFuturesGetInstrumentsTool(),
        KrakenFuturesGetInstrumentStatusTool(),
        KrakenFuturesGetInstrumentStatusListTool(),
    ]


@lru_cache
def get_futures_trading_tools() -> list:
    """Get trading/order management tools (requires auth)."""
    creds = _get_credentials()
    return [
        KrakenFuturesSendOrderTool(**creds),
        KrakenFuturesEditOrderTool(**creds),
        KrakenFuturesCancelOrderTool(**creds),
        KrakenFuturesCancelAllOrdersTool(**creds),
        KrakenFuturesBatchOrderTool(**creds),
        KrakenFuturesGetOpenOrdersTool(**creds),
        KrakenFuturesGetOrderStatusTool(**creds),
        KrakenFuturesDeadMansSwitchTool(**creds),
    ]


@lru_cache
def get_futures_account_tools() -> list:
    """Get account management tools (requires auth)."""
    creds = _get_credentials()
    return [
        KrakenFuturesGetWalletsTool(**creds),
        KrakenFuturesGetOpenPositionsTool(**creds),
        KrakenFuturesGetPortfolioMarginTool(**creds),
        KrakenFuturesCalculateMarginPnLTool(**creds),
        KrakenFuturesGetUnwindQueueTool(**creds),
        KrakenFuturesGetLeverageTool(**creds),
        KrakenFuturesSetLeverageTool(**creds),
        KrakenFuturesGetPnLCurrencyTool(**creds),
        KrakenFuturesSetPnLCurrencyTool(**creds),
    ]


@lru_cache
def get_futures_transfer_tools() -> list:
    """Get transfer/funding tools (requires auth)."""
    creds = _get_credentials()
    return [
        KrakenFuturesWalletTransferTool(**creds),
        KrakenFuturesSubAccountTransferTool(**creds),
        KrakenFuturesWithdrawToSpotTool(**creds),
    ]


@lru_cache
def get_futures_subaccount_tools() -> list:
    """Get subaccount management tools (requires auth)."""
    creds = _get_credentials()
    return [
        KrakenFuturesGetSubaccountsTool(**creds),
        KrakenFuturesCheckTradingStatusTool(**creds),
        KrakenFuturesUpdateTradingStatusTool(**creds),
    ]


@lru_cache
def get_futures_fee_tools() -> list:
    """Get fee schedule tools (requires auth)."""
    creds = _get_credentials()
    return [
        KrakenFuturesGetFeeSchedulesTool(**creds),
        KrakenFuturesGetFeeVolumesTool(**creds),
    ]


@lru_cache
def get_futures_assignment_tools() -> list:
    """Get assignment program tools (requires auth)."""
    creds = _get_credentials()
    return [
        KrakenFuturesListAssignmentProgramsTool(**creds),
        KrakenFuturesAddAssignmentPreferenceTool(**creds),
        KrakenFuturesDeleteAssignmentPreferenceTool(**creds),
        KrakenFuturesListAssignmentHistoryTool(**creds),
    ]


@lru_cache
def get_futures_historical_tools() -> list:
    """Get historical data tools."""
    creds = _get_credentials()
    return [
        KrakenFuturesGetFillsTool(**creds),
        KrakenFuturesGetFundingRatesTool(),  # Public
    ]


@lru_cache
def get_futures_all_tools() -> list:
    """Get all Futures tools."""
    return (
        get_futures_market_tools()
        + get_futures_trading_tools()
        + get_futures_account_tools()
        + get_futures_transfer_tools()
        + get_futures_subaccount_tools()
        + get_futures_fee_tools()
        + get_futures_assignment_tools()
        + get_futures_historical_tools()
    )


# =============================================================================
# Role-based tool sets (for specific agent roles)
# =============================================================================


@lru_cache
def get_futures_leadership_tools() -> list:
    """Tools for Futures leadership agents (CIO, Head, CRO, COO).

    Focus on oversight: wallets, positions, margin, PnL.
    """
    creds = _get_credentials()
    return [
        # Market overview
        KrakenFuturesGetTickersTool(),
        KrakenFuturesGetInstrumentsTool(),
        KrakenFuturesGetInstrumentStatusListTool(),
        # Account overview
        KrakenFuturesGetWalletsTool(**creds),
        KrakenFuturesGetOpenPositionsTool(**creds),
        KrakenFuturesGetPortfolioMarginTool(**creds),
        KrakenFuturesCalculateMarginPnLTool(**creds),
        KrakenFuturesGetOpenOrdersTool(**creds),
        KrakenFuturesGetLeverageTool(**creds),
        # Notifications
        KrakenFuturesGetNotificationsTool(**creds),
    ]


@lru_cache
def get_futures_research_tools() -> list:
    """Tools for Futures research/analyst agents.

    Focus on market data, funding rates, instruments.
    """
    return [
        # Market data
        KrakenFuturesGetTickersTool(),
        KrakenFuturesGetTickerTool(),
        KrakenFuturesGetOrderBookTool(),
        KrakenFuturesGetTradeHistoryTool(),
        # Instruments
        KrakenFuturesGetInstrumentsTool(),
        KrakenFuturesGetInstrumentStatusTool(),
        KrakenFuturesGetInstrumentStatusListTool(),
        # Funding rates (important for research)
        KrakenFuturesGetFundingRatesTool(),
    ]


@lru_cache
def get_futures_execution_tools() -> list:
    """Tools for Futures execution/trading agents.

    Focus on order management.
    """
    creds = _get_credentials()
    return [
        # Market data for execution
        KrakenFuturesGetTickersTool(),
        KrakenFuturesGetOrderBookTool(),
        KrakenFuturesGetInstrumentStatusTool(),
        # Order management
        KrakenFuturesSendOrderTool(**creds),
        KrakenFuturesEditOrderTool(**creds),
        KrakenFuturesCancelOrderTool(**creds),
        KrakenFuturesCancelAllOrdersTool(**creds),
        KrakenFuturesBatchOrderTool(**creds),
        KrakenFuturesGetOpenOrdersTool(**creds),
        KrakenFuturesGetOrderStatusTool(**creds),
        KrakenFuturesDeadMansSwitchTool(**creds),
        # Position info
        KrakenFuturesGetOpenPositionsTool(**creds),
    ]


@lru_cache
def get_futures_risk_tools() -> list:
    """Tools for Futures risk management agents.

    Focus on positions, margin, leverage, and emergency actions.
    """
    creds = _get_credentials()
    return [
        # Market data
        KrakenFuturesGetTickersTool(),
        KrakenFuturesGetInstrumentStatusListTool(),
        # Position/Margin monitoring
        KrakenFuturesGetWalletsTool(**creds),
        KrakenFuturesGetOpenPositionsTool(**creds),
        KrakenFuturesGetPortfolioMarginTool(**creds),
        KrakenFuturesCalculateMarginPnLTool(**creds),
        KrakenFuturesGetUnwindQueueTool(**creds),
        # Leverage management
        KrakenFuturesGetLeverageTool(**creds),
        KrakenFuturesSetLeverageTool(**creds),
        # Emergency actions
        KrakenFuturesCancelAllOrdersTool(**creds),
        KrakenFuturesDeadMansSwitchTool(**creds),
        # Order monitoring
        KrakenFuturesGetOpenOrdersTool(**creds),
    ]


@lru_cache
def get_futures_operations_tools() -> list:
    """Tools for Futures operations agents (Controller, Treasury, Security, Compliance, Ops).

    Focus on transfers, fees, subaccounts, and compliance.
    """
    creds = _get_credentials()
    return [
        # Account
        KrakenFuturesGetWalletsTool(**creds),
        KrakenFuturesGetOpenPositionsTool(**creds),
        KrakenFuturesGetPortfolioMarginTool(**creds),
        # Transfers
        KrakenFuturesWalletTransferTool(**creds),
        KrakenFuturesSubAccountTransferTool(**creds),
        KrakenFuturesWithdrawToSpotTool(**creds),
        # Subaccounts
        KrakenFuturesGetSubaccountsTool(**creds),
        KrakenFuturesCheckTradingStatusTool(**creds),
        KrakenFuturesUpdateTradingStatusTool(**creds),
        # Fees
        KrakenFuturesGetFeeSchedulesTool(**creds),
        KrakenFuturesGetFeeVolumesTool(**creds),
        # History
        KrakenFuturesGetFillsTool(**creds),
        # Notifications
        KrakenFuturesGetNotificationsTool(**creds),
        # PnL settings
        KrakenFuturesGetPnLCurrencyTool(**creds),
        KrakenFuturesSetPnLCurrencyTool(**creds),
    ]
