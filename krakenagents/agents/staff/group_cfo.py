"""STAFF-04: Group CFO Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import (
    PnLTrackerTool,
)
from krakenagents.tools import (
    get_spot_operations_tools,
    get_futures_operations_tools,
)


def create_group_cfo_agent() -> Agent:
    """Create STAFF-04 Group CFO Agent.

    The Group CFO is responsible for:
    - Correct PnL/NAV calculation and attribution
    - Cost control and management reporting
    - Performance attribution per desk/pod/strategy
    - Fee and trading cost optimization

    Reports to: STAFF-00 (CEO)
    Uses light LLM for financial operations.
    """
    # Combine operations tools from both desks for financial data
    tools = [
        PnLTrackerTool(),
    ] + get_spot_operations_tools() + get_futures_operations_tools()

    return create_light_agent(
        role="Group CFO — PnL, NAV, Attribution, and Cost Control",
        goal=(
            "Ensure correct PnL/NAV/attribution, control costs, and provide management reporting. "
            "Run daily NAV/PnL consolidation across spot and derivatives. Provide performance "
            "attribution per desk/pod/strategy. Maintain cost dashboards and track budget variances. "
            "Identify top vs weak strategies for allocation decisions. Optimize trading costs "
            "(fees/funding/borrow) to maximize net PnL."
        ),
        backstory=(
            "Finance professional with deep expertise in fund accounting, performance attribution, "
            "and trading cost analysis. Strong understanding of crypto-specific P&L components "
            "(funding rates, borrow costs, liquidation fees). Known for accurate and timely "
            "financial reporting. Expert in identifying cost optimization opportunities and "
            "tracking fee tier improvements."
        ),
        tools=tools,
        allow_delegation=False,
    )
