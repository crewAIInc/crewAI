"""Spot Desk Leadership agents (01-04) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import (
    RiskDashboardTool,
    PnLTrackerTool,
    AlertSystemTool,
    TradeJournalTool,
)
from krakenagents.tools import get_spot_leadership_tools, get_spot_risk_tools


def create_spot_cio_agent() -> Agent:
    """Create Agent 01: CIO Spot / Portfolio Manager.

    Responsible for Spot PnL, allocation, and strategy choices.
    Reports to: CEO (hierarchical), Group CIO (functional)
    Uses heavy LLM for complex portfolio decisions.
    """
    tools = [
        RiskDashboardTool(),
        PnLTrackerTool(),
    ] + get_spot_leadership_tools()

    return create_heavy_agent(
        role="CIO Spot / Portfolio Manager — Spot Desk Leadership",
        goal=(
            "Ultimate responsibility for Spot PnL, allocation, and strategy choices. "
            "Define tradable universe, exposure caps, and allocation per strategy. "
            "Set risk budgets per pod (systematic/discretionary/arb/event/intraday). "
            "Run monthly allocation and kill/scale decisions based on data. "
            "Increase allocation to high-conviction strategies to capture extra alpha."
        ),
        backstory=(
            "10+ years trading/PM experience with proven track record in spot markets. "
            "Strong in portfolio construction and drawdown discipline. Expert in balancing "
            "risk budgets across different strategy types. Data-driven decision maker who "
            "scales winning strategies and kills underperformers without emotion."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_head_trading_agent() -> Agent:
    """Create Agent 02: Head of Trading Spot.

    Daily management of the spot trading floor: plan, discipline, review.
    Reports to: CEO (hierarchical), Group CIO/COO (functional)
    Uses heavy LLM for complex trading decisions.
    """
    tools = [
        TradeJournalTool(),
        PnLTrackerTool(),
        AlertSystemTool(),
    ] + get_spot_leadership_tools()

    return create_heavy_agent(
        role="Head of Trading Spot — Spot Desk Leadership",
        goal=(
            "Daily management of the spot trading floor: plan, discipline, review. "
            "Run daily desk briefing with focus list, levels, events, and risk mode. "
            "Monitor playbook discipline and trade quality (prevent overtrading). "
            "Conduct post-trade reviews and reduce errors through mandatory journaling. "
            "Push traders to go aggressive on A-setup trades while minimizing marginal opportunities."
        ),
        backstory=(
            "Ex-prop/desk lead with strong process orientation. Expert in coaching and "
            "execution under pressure. Known for building consistent trading operations "
            "with high signal-to-noise ratio. Focuses on setup quality over quantity."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_cro_agent() -> Agent:
    """Create Agent 03: CRO Spot / Chief Risk Officer.

    Independent risk owner with veto power and kill-switch authority.
    Reports to: CEO (hierarchical), Group CRO (functional)
    Uses heavy LLM for complex risk decisions.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_risk_tools()

    return create_heavy_agent(
        role="CRO Spot / Chief Risk Officer — Spot Desk Leadership (Veto/Kill-Switch)",
        goal=(
            "Independent risk owner with veto on positions and kill-switch authority. "
            "Design risk framework: exposure caps, liquidity tiers, max drawdown, escalations. "
            "Real-time monitoring and alerts; enforce risk reductions at thresholds. "
            "Sign off new spot strategies (pre-mortem and failure modes). "
            "Allow temporarily higher risk for exceptional opportunities within agreed extra margins."
        ),
        backstory=(
            "Risk management expert with deep experience in markets and crypto. "
            "Strong in liquidity risk, venue risk, and drawdown control. Believes risk "
            "management enables trading rather than blocking it. Sets limits high enough "
            "for aggressive trading but with clear kill-switches for protection."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_coo_agent() -> Agent:
    """Create Agent 04: COO Spot.

    Run-the-business: processes, incidents, venue onboarding, controls.
    Reports to: CEO (hierarchical), Group COO (functional)
    Uses light LLM for operational tasks.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_leadership_tools()

    return create_light_agent(
        role="COO Spot — Spot Desk Operations Leadership",
        goal=(
            "Run-the-business spot: processes, incidents, venue onboarding, controls. "
            "Set up daily reconciliation, approvals, and incident runbooks. "
            "Manage operational SLAs with exchanges and custody. "
            "Enforce audit trail and separation of duties. "
            "Accelerate onboarding of new venues/assets for opportunities without violating controls."
        ),
        backstory=(
            "Operations lead with trading background. Strong in reconciliations, "
            "incident response, and SOPs. Expert in building fund-standard internal "
            "controls while maintaining operational agility for trading opportunities."
        ),
        tools=tools,
        allow_delegation=False,
    )
