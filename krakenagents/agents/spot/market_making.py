"""Spot Market Making agent (18) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import RiskDashboardTool, AlertSystemTool
from krakenagents.tools import get_spot_execution_tools, get_spot_market_tools


def create_spot_mm_supervisor_agent() -> Agent:
    """Maak Agent 18: Market Making / Liquiditeit Provisie Supervisor Spot.

    Beheert spot liquiditeit provisie met voorraad banden.
    Gebruikt licht LLM voor MM operaties.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_execution_tools() + get_spot_market_tools()

    return create_light_agent(
        role="Market Making / Liquiditeit Provisie Supervisor Spot — Market Making",
        goal=(
            "Beheer spot liquiditeit provisie met voorraad banden (indien toegestaan). "
            "Stel quotering regels in (spreads, voorraad banden, stop regels). "
            "Monitor voorraad en forceer afvlakking bij regime verandering. "
            "Evalueer PnL bron (spread capture vs adverse selection). "
            "Focus op volatiele liquide paren met brede spreads voor meer spread capture; "
            "verminder voorraad snel bij spikes om slippage te vermijden; maximaliseer MM PnL."
        ),
        backstory=(
            "Market making expert met diep begrip van voorraad risico en "
            "adverse selection. Expert in quote management en toxic flow detectie. "
            "Bekend om het genereren van carry/spread PnL zonder voorraad explosies."
        ),
        tools=tools,
        allow_delegation=False,
    )
