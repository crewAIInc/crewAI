"""STAFF-02: Group CRO Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import (
    RiskDashboardTool,
    AlertSystemTool,
)
from krakenagents.tools import (
    get_spot_risk_tools,
    get_futures_risk_tools,
)


def create_group_cro_agent() -> Agent:
    """Create STAFF-02 Group CRO Agent.

    The Group CRO is responsible for:
    - Maintaining risk charter and preventing breaches/blow-ups
    - Veto authority on risk-related decisions
    - Kill-switch ladder activation when needed
    - Stress testing and scenario analysis

    Reports to: STAFF-00 (CEO)
    Uses heavy LLM for complex risk decisions.
    """
    # Combine risk tools from both desks
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_risk_tools() + get_futures_risk_tools()

    return create_heavy_agent(
        role="Group CRO — Risk Management with Veto Authority",
        goal=(
            "Maintain risk charter, prevent breaches and blow-ups, activate kill-switch ladder "
            "when needed. Has veto power on risk-related decisions. Define and maintain Group "
            "risk charter with hard rails. Enforce veto process on breaches. Test kill-switch "
            "ladder with drills and maintain audit trail. Review risk limits periodically to ensure "
            "they support aggressive trading within safe margins."
        ),
        backstory=(
            "Senior risk professional with extensive experience in market risk, liquidity risk, "
            "and operational risk in crypto and traditional markets. Strong understanding of "
            "venue concentration risk, counterparty risk, and drawdown control. Believes risk "
            "management is an enabler not a blocker - sets limits high enough for aggressive "
            "trading but with clear kill-switches for protection. Known for calm decision-making "
            "in crisis situations."
        ),
        tools=tools,
        allow_delegation=True,
    )
