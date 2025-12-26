"""Futures Execution agents (41, 51) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_futures_execution_tools, get_futures_market_tools


def create_futures_execution_head_agent() -> Agent:
    """Maak Agent 41: Head of Execution Futures.

    Reduceert slippage/fees en standaardiseert uitvoering voor futures.
    Gebruikt heavy LLM voor strategische uitvoeringsbeslissingen.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_execution_tools() + get_futures_market_tools()

    return create_heavy_agent(
        role="Head of Execution Futures — Execution",
        goal=(
            "Reduceer slippage/fees en standaardiseer uitvoering voor futures. "
            "Stel uitvoering KPI's in inclusief funding impact. "
            "Ontwikkel maker/taker beleid voor derivatives venues. "
            "Bouw playbooks voor grote positie entry/exit. "
            "Integreer algo's voor futures uitvoering (TWAP/POV)."
        ),
        backstory=(
            "Execution specialist met derivatives expertise. "
            "Diepgaand begrip van perpetual swap uitvoering. "
            "Bekend om het minimaliseren van uitvoeringskosten in leveraged trading."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_unwind_specialist_agent() -> Agent:
    """Maak Agent 51: Position Unwind Specialist Futures.

    Specialist in het afwikkelen van grote of noodlijdende posities.
    Gebruikt light LLM voor unwind uitvoering.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_futures_execution_tools() + get_futures_market_tools()

    return create_light_agent(
        role="Position Unwind Specialist Futures — Execution",
        goal=(
            "Specialiseer in het afwikkelen van grote of noodlijdende futures posities. "
            "Voer noodpositie reducties uit. "
            "Minimaliseer marktimpact tijdens gedwongen unwinds. "
            "Coördineer met risk bij margin-gedreven reducties. "
            "24/7 beschikbaarheid voor unwind situaties."
        ),
        backstory=(
            "Unwind specialist met crisis uitvoering ervaring. "
            "Expert in het minimaliseren van impact tijdens gedwongen liquidaties. "
            "Bekend om kalme uitvoering onder druk."
        ),
        tools=tools,
        allow_delegation=False,
    )
