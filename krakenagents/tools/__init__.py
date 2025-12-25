"""Tools module for QRI Trading Organization.

Provides Kraken Spot and Futures API tools with proper credential configuration.
Also provides crew delegation tools for hierarchical management.
"""

from krakenagents.tools.kraken_spot import (
    get_spot_market_tools,
    get_spot_trading_tools,
    get_spot_account_tools,
    get_spot_funding_tools,
    get_spot_earn_tools,
    get_spot_all_tools,
    get_spot_leadership_tools,
    get_spot_research_tools,
    get_spot_execution_tools,
    get_spot_risk_tools,
    get_spot_operations_tools,
)
from krakenagents.tools.kraken_futures import (
    get_futures_market_tools,
    get_futures_trading_tools,
    get_futures_account_tools,
    get_futures_all_tools,
    get_futures_leadership_tools,
    get_futures_research_tools,
    get_futures_execution_tools,
    get_futures_risk_tools,
    get_futures_operations_tools,
)
from krakenagents.tools.crew_delegation import (
    get_delegation_tools,
    DelegateToSpotDeskTool,
    DelegateToFuturesDeskTool,
    GetDeskStatusTool,
    DelegateToBothDesksTool,
    set_crew_manager,
)

__all__ = [
    # Spot tools
    "get_spot_market_tools",
    "get_spot_trading_tools",
    "get_spot_account_tools",
    "get_spot_funding_tools",
    "get_spot_earn_tools",
    "get_spot_all_tools",
    "get_spot_leadership_tools",
    "get_spot_research_tools",
    "get_spot_execution_tools",
    "get_spot_risk_tools",
    "get_spot_operations_tools",
    # Futures tools
    "get_futures_market_tools",
    "get_futures_trading_tools",
    "get_futures_account_tools",
    "get_futures_all_tools",
    "get_futures_leadership_tools",
    "get_futures_research_tools",
    "get_futures_execution_tools",
    "get_futures_risk_tools",
    "get_futures_operations_tools",
    # Delegation tools
    "get_delegation_tools",
    "DelegateToSpotDeskTool",
    "DelegateToFuturesDeskTool",
    "GetDeskStatusTool",
    "DelegateToBothDesksTool",
    "set_crew_manager",
]
