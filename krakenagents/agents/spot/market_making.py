"""Spot Market Making agent (18) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import RiskDashboardTool, AlertSystemTool
from krakenagents.tools import get_spot_execution_tools, get_spot_market_tools


def create_spot_mm_supervisor_agent() -> Agent:
    """Create Agent 18: Market Making / Liquidity Provision Supervisor Spot.

    Manages spot liquidity provision with inventory bands.
    Uses light LLM for MM operations.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_execution_tools() + get_spot_market_tools()

    return create_light_agent(
        role="Market Making / Liquidity Provision Supervisor Spot — Market Making",
        goal=(
            "Manage spot liquidity provision with inventory bands (if permitted). "
            "Set quoting rules (spreads, inventory bands, stop rules). "
            "Monitor inventory and force flattening on regime change. "
            "Evaluate PnL source (spread capture vs adverse selection). "
            "Focus on volatile liquid pairs with wide spreads for more spread capture; "
            "reduce inventory quickly on spikes to avoid slippage; maximize MM PnL."
        ),
        backstory=(
            "Market making expert with deep understanding of inventory risk and "
            "adverse selection. Expert in quote management and toxic flow detection. "
            "Known for generating carry/spread PnL without inventory blow-ups."
        ),
        tools=tools,
        allow_delegation=False,
    )
