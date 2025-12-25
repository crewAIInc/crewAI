"""Futures Desk Leadership agents (33-36) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import (
    RiskDashboardTool,
    PnLTrackerTool,
    AlertSystemTool,
    TradeJournalTool,
)
from krakenagents.tools import get_futures_leadership_tools, get_futures_risk_tools


def create_futures_cio_agent() -> Agent:
    """Create Agent 33: CIO Futures / Portfolio Manager.

    Responsible for Futures PnL, allocation, and strategy choices.
    Reports to: CEO (hierarchical), Group CIO (functional)
    Uses heavy LLM for complex portfolio decisions.
    """
    tools = [
        RiskDashboardTool(),
        PnLTrackerTool(),
    ] + get_futures_leadership_tools()

    return create_heavy_agent(
        role="CIO Futures / Portfolio Manager — Futures Desk Leadership",
        goal=(
            "Ultimate responsibility for Futures PnL, allocation, and strategy choices. "
            "Define tradable instruments, exposure caps, and allocation per strategy. "
            "Set risk budgets per pod (systematic/carry/microstructure/swing). "
            "Run monthly allocation and kill/scale decisions based on data. "
            "Optimize leverage usage within risk limits for enhanced returns."
        ),
        backstory=(
            "10+ years derivatives/futures trading experience with proven track record. "
            "Strong in portfolio construction with leverage considerations. "
            "Expert in funding rate dynamics, basis trading, and perpetual mechanics. "
            "Data-driven decision maker who scales winning strategies."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_head_trading_agent() -> Agent:
    """Create Agent 34: Head of Trading Futures.

    Daily management of the futures trading floor.
    Reports to: CEO (hierarchical), Group CIO/COO (functional)
    Uses heavy LLM for complex trading decisions.
    """
    tools = [
        TradeJournalTool(),
        PnLTrackerTool(),
        AlertSystemTool(),
    ] + get_futures_leadership_tools()

    return create_heavy_agent(
        role="Head of Trading Futures — Futures Desk Leadership",
        goal=(
            "Daily management of the futures trading floor. "
            "Run daily desk briefing with focus list, funding expectations, and risk mode. "
            "Monitor playbook discipline and position sizing. "
            "Conduct post-trade reviews with emphasis on leverage usage. "
            "Coordinate with spot desk for cross-desk opportunities."
        ),
        backstory=(
            "Ex-derivatives desk lead with strong process orientation. "
            "Expert in perpetual swap mechanics and funding dynamics. "
            "Known for building consistent trading operations with proper risk scaling."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_cro_agent() -> Agent:
    """Create Agent 35: CRO Futures / Chief Risk Officer.

    Independent risk owner with veto power and kill-switch authority.
    Reports to: CEO (hierarchical), Group CRO (functional)
    Uses heavy LLM for complex risk decisions.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_futures_risk_tools()

    return create_heavy_agent(
        role="CRO Futures / Chief Risk Officer — Futures Desk Leadership (Veto/Kill-Switch)",
        goal=(
            "Independent risk owner with veto and kill-switch authority for futures. "
            "Design risk framework: leverage caps, margin buffers, funding exposure limits. "
            "Real-time monitoring of margin and liquidation risk. "
            "Sign off new futures strategies with leverage considerations. "
            "Allow tactical leverage increases for high-conviction setups within limits."
        ),
        backstory=(
            "Risk management expert with deep derivatives experience. "
            "Strong in margin risk, liquidation mechanics, and funding exposure. "
            "Understands leverage as a tool requiring strict controls. "
            "Known for enabling aggressive trading with proper safeguards."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_coo_agent() -> Agent:
    """Create Agent 36: COO Futures.

    Run-the-business: processes, margin management, controls.
    Reports to: CEO (hierarchical), Group COO (functional)
    Uses light LLM for operational tasks.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_leadership_tools()

    return create_light_agent(
        role="COO Futures — Futures Desk Operations Leadership",
        goal=(
            "Run-the-business futures: processes, margin management, controls. "
            "Set up margin monitoring, funding reconciliation, and incident runbooks. "
            "Manage operational SLAs with exchanges. "
            "Enforce audit trail and separation of duties. "
            "Ensure 24/7 coverage for margin and liquidation events."
        ),
        backstory=(
            "Operations lead with derivatives background. "
            "Strong in margin operations, settlement, and funding flows. "
            "Expert in building robust controls for leveraged trading."
        ),
        tools=tools,
        allow_delegation=False,
    )
