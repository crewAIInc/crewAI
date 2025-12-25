"""Kraken Spot API tools for QRI Trading Organization.

Provides tool factory functions that return configured tool instances
with API credentials from settings.
"""

from functools import lru_cache

from crewai.tools.kraken import (
    # Market Data (9) - PUBLIC
    GetServerTimeTool,
    GetSystemStatusTool,
    GetAssetInfoTool,
    GetTradableAssetPairsTool,
    GetTickerInformationTool,
    GetOrderBookTool,
    GetRecentTradesTool,
    GetRecentSpreadsTool,
    GetOHLCDataTool,
    # Trading (9) - PRIVATE
    AddOrderTool,
    AddOrderBatchTool,
    AmendOrderTool,
    EditOrderTool,
    CancelOrderTool,
    CancelOrderBatchTool,
    CancelAllOrdersTool,
    CancelAllOrdersAfterXTool,
    GetWebSocketsTokenTool,
    # Account (17) - PRIVATE
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
    # Funding (10) - PRIVATE
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
    # Earn (6) - PRIVATE
    ListEarnStrategiesTool,
    ListEarnAllocationsTool,
    AllocateEarnFundsTool,
    GetAllocationStatusTool,
    DeallocateEarnFundsTool,
    GetDeallocationStatusTool,
    # Subaccounts (2) - PRIVATE
    CreateSubaccountTool,
    AccountTransferTool,
    # Transparency (2) - PUBLIC
    GetPreTradeDataTool,
    GetPostTradeDataTool,
)

from krakenagents.config import get_settings


def _get_credentials() -> dict[str, str]:
    """Get Spot API credentials from settings."""
    settings = get_settings()
    return {
        "api_key": settings.kraken_api_key,
        "api_secret": settings.kraken_api_secret,
    }


# =============================================================================
# Category-based tool getters (for specific agent types)
# =============================================================================


@lru_cache
def get_spot_market_tools() -> list:
    """Get public market data tools (no auth required)."""
    return [
        GetServerTimeTool(),
        GetSystemStatusTool(),
        GetAssetInfoTool(),
        GetTradableAssetPairsTool(),
        GetTickerInformationTool(),
        GetOrderBookTool(),
        GetRecentTradesTool(),
        GetRecentSpreadsTool(),
        GetOHLCDataTool(),
    ]


@lru_cache
def get_spot_trading_tools() -> list:
    """Get trading tools (requires auth)."""
    creds = _get_credentials()
    return [
        AddOrderTool(**creds),
        AddOrderBatchTool(**creds),
        AmendOrderTool(**creds),
        EditOrderTool(**creds),
        CancelOrderTool(**creds),
        CancelOrderBatchTool(**creds),
        CancelAllOrdersTool(**creds),
        CancelAllOrdersAfterXTool(**creds),
        GetWebSocketsTokenTool(**creds),
    ]


@lru_cache
def get_spot_account_tools() -> list:
    """Get account tools (requires auth)."""
    creds = _get_credentials()
    return [
        GetAccountBalanceTool(**creds),
        GetExtendedBalanceTool(**creds),
        GetTradeBalanceTool(**creds),
        GetOpenOrdersTool(**creds),
        GetClosedOrdersTool(**creds),
        QueryOrdersInfoTool(**creds),
        GetTradesHistoryTool(**creds),
        QueryTradesInfoTool(**creds),
        GetOpenPositionsTool(**creds),
        GetLedgersTool(**creds),
        QueryLedgersTool(**creds),
        GetTradeVolumeTool(**creds),
        RequestExportReportTool(**creds),
        GetExportReportStatusTool(**creds),
        RetrieveDataExportTool(**creds),
        DeleteExportReportTool(**creds),
        GetOrderAmendsTool(**creds),
    ]


@lru_cache
def get_spot_funding_tools() -> list:
    """Get funding tools (requires auth)."""
    creds = _get_credentials()
    return [
        GetDepositMethodsTool(**creds),
        GetDepositAddressesTool(**creds),
        GetDepositStatusTool(**creds),
        GetWithdrawalMethodsTool(**creds),
        GetWithdrawalAddressesTool(**creds),
        GetWithdrawalInfoTool(**creds),
        WithdrawFundsTool(**creds),
        GetWithdrawalStatusTool(**creds),
        WalletTransferTool(**creds),
        CancelWithdrawalTool(**creds),
    ]


@lru_cache
def get_spot_earn_tools() -> list:
    """Get earn/staking tools (requires auth)."""
    creds = _get_credentials()
    return [
        ListEarnStrategiesTool(**creds),
        ListEarnAllocationsTool(**creds),
        AllocateEarnFundsTool(**creds),
        GetAllocationStatusTool(**creds),
        DeallocateEarnFundsTool(**creds),
        GetDeallocationStatusTool(**creds),
    ]


@lru_cache
def get_spot_subaccount_tools() -> list:
    """Get subaccount management tools (requires auth)."""
    creds = _get_credentials()
    return [
        CreateSubaccountTool(**creds),
        AccountTransferTool(**creds),
    ]


@lru_cache
def get_spot_transparency_tools() -> list:
    """Get transparency/regulatory tools (public)."""
    return [
        GetPreTradeDataTool(),
        GetPostTradeDataTool(),
    ]


@lru_cache
def get_spot_all_tools() -> list:
    """Get all Spot tools."""
    return (
        get_spot_market_tools()
        + get_spot_trading_tools()
        + get_spot_account_tools()
        + get_spot_funding_tools()
        + get_spot_earn_tools()
        + get_spot_subaccount_tools()
        + get_spot_transparency_tools()
    )


# =============================================================================
# Role-based tool sets (for specific agent roles)
# =============================================================================


@lru_cache
def get_spot_leadership_tools() -> list:
    """Tools for Spot leadership agents (CIO, Head, CRO, COO).

    Focus on oversight: balances, positions, trade history.
    """
    creds = _get_credentials()
    return [
        # Market overview
        GetSystemStatusTool(),
        GetTickerInformationTool(),
        GetAssetInfoTool(),
        # Account overview
        GetAccountBalanceTool(**creds),
        GetTradeBalanceTool(**creds),
        GetOpenPositionsTool(**creds),
        GetOpenOrdersTool(**creds),
        GetTradesHistoryTool(**creds),
        GetTradeVolumeTool(**creds),
    ]


@lru_cache
def get_spot_research_tools() -> list:
    """Tools for Spot research/analyst agents.

    Focus on market data analysis.
    """
    return [
        # Market data
        GetServerTimeTool(),
        GetSystemStatusTool(),
        GetAssetInfoTool(),
        GetTradableAssetPairsTool(),
        GetTickerInformationTool(),
        GetOrderBookTool(),
        GetRecentTradesTool(),
        GetRecentSpreadsTool(),
        GetOHLCDataTool(),
        # Transparency
        GetPreTradeDataTool(),
        GetPostTradeDataTool(),
    ]


@lru_cache
def get_spot_execution_tools() -> list:
    """Tools for Spot execution/trading agents.

    Focus on order management.
    """
    creds = _get_credentials()
    return [
        # Market data for execution
        GetTickerInformationTool(),
        GetOrderBookTool(),
        GetRecentTradesTool(),
        # Trading
        AddOrderTool(**creds),
        AddOrderBatchTool(**creds),
        AmendOrderTool(**creds),
        EditOrderTool(**creds),
        CancelOrderTool(**creds),
        CancelOrderBatchTool(**creds),
        CancelAllOrdersTool(**creds),
        CancelAllOrdersAfterXTool(**creds),
        # Order status
        GetOpenOrdersTool(**creds),
        GetClosedOrdersTool(**creds),
        QueryOrdersInfoTool(**creds),
    ]


@lru_cache
def get_spot_risk_tools() -> list:
    """Tools for Spot risk management agents.

    Focus on positions, balances, and emergency actions.
    """
    creds = _get_credentials()
    return [
        # Market data
        GetTickerInformationTool(),
        GetSystemStatusTool(),
        # Account/Position monitoring
        GetAccountBalanceTool(**creds),
        GetExtendedBalanceTool(**creds),
        GetTradeBalanceTool(**creds),
        GetOpenPositionsTool(**creds),
        GetOpenOrdersTool(**creds),
        # Emergency actions
        CancelAllOrdersTool(**creds),
        CancelAllOrdersAfterXTool(**creds),
        # History for analysis
        GetTradesHistoryTool(**creds),
        GetLedgersTool(**creds),
    ]


@lru_cache
def get_spot_operations_tools() -> list:
    """Tools for Spot operations agents (Controller, Treasury, Security, Compliance, Ops).

    Focus on reporting, funding, and compliance.
    """
    creds = _get_credentials()
    return [
        # Account
        GetAccountBalanceTool(**creds),
        GetExtendedBalanceTool(**creds),
        GetTradeBalanceTool(**creds),
        GetLedgersTool(**creds),
        QueryLedgersTool(**creds),
        GetTradeVolumeTool(**creds),
        # Reporting
        RequestExportReportTool(**creds),
        GetExportReportStatusTool(**creds),
        RetrieveDataExportTool(**creds),
        DeleteExportReportTool(**creds),
        # Funding
        GetDepositMethodsTool(**creds),
        GetDepositAddressesTool(**creds),
        GetDepositStatusTool(**creds),
        GetWithdrawalMethodsTool(**creds),
        GetWithdrawalAddressesTool(**creds),
        GetWithdrawalStatusTool(**creds),
        WalletTransferTool(**creds),
        # Subaccounts
        CreateSubaccountTool(**creds),
        AccountTransferTool(**creds),
        # Transparency/Compliance
        GetPreTradeDataTool(),
        GetPostTradeDataTool(),
    ]
