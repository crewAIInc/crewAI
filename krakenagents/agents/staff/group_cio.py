"""STAFF-01: Group CIO Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import (
    RiskDashboardTool,
    PnLTrackerTool,
)
from krakenagents.tools import (
    get_spot_leadership_tools,
    get_futures_leadership_tools,
)
from krakenagents.tools.crew_delegation import (
    DelegateToSpotDeskTool,
    DelegateToFuturesDeskTool,
    GetDeskStatusTool,
    DelegateToBothDesksTool,
)


def create_group_cio_agent() -> Agent:
    """Create STAFF-01 Group CIO Agent.

    The Group CIO is responsible for:
    - Orchestrating portfolio/allocation across Spot + Derivatives desks
    - Consistent allocation process
    - Kill/scale reviews based on data
    - Identifying high-alpha opportunities
    - DELEGATING tasks to Spot and Futures trading desks

    Reports to: STAFF-00 (CEO)
    Uses heavy LLM for complex portfolio decisions.
    """
    # Combine leadership tools from both desks + delegation tools
    tools = [
        RiskDashboardTool(),
        PnLTrackerTool(),
        # Delegation tools - allows CIO to control trading desks
        DelegateToSpotDeskTool(),
        DelegateToFuturesDeskTool(),
        GetDeskStatusTool(),
        DelegateToBothDesksTool(),
    ] + get_spot_leadership_tools() + get_futures_leadership_tools()

    return create_heavy_agent(
        role="Group CIO — Portfolio/Allocation across Spot + Derivatives Desks",
        goal=(
            "Orchestrate portfolio allocation across Spot and Derivatives desks with consistent "
            "allocation process. Collect desk proposals, create allocation recommendations, "
            "run monthly kill/scale reviews data-first, and identify high-alpha opportunities "
            "for pilot capital allocation. "
            "IMPORTANT: You can delegate trading tasks to the Spot desk (32 agents) and "
            "Futures desk (32 agents) using the delegation tools."
        ),
        backstory=(
            "Experienced CIO with deep expertise in portfolio construction, factor analysis, "
            "and cross-asset allocation. Strong quantitative background with ability to balance "
            "risk budgets across different strategy types. Believes in data-driven decisions "
            "and systematic review processes. Can identify capacity constraints and slippage "
            "implications for scaling strategies. "
            "As Group CIO, you have the authority to delegate tasks to the Spot and Futures trading desks."
        ),
        tools=tools,
        allow_delegation=True,
    )
