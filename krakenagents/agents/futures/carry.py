"""Futures Carry/Funding Trading agents (38, 43, 44) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_futures_research_tools, get_futures_execution_tools


def create_futures_carry_head_agent() -> Agent:
    """Maak Agent 38: Head of Carry/Funding Futures.

    Eigenaar van funding rate en carry strategieën.
    Gebruikt heavy LLM voor strategische carry beslissingen.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Head of Carry/Funding Futures — Carry Trading",
        goal=(
            "Eigenaar funding rate en carry strategieën. "
            "Ontwerp funding capture strategieën over venues. "
            "Monitor cross-exchange funding differentiëlen. "
            "Beheer basis posities voor yield extractie. "
            "Schaal carry strategieën in gunstige funding regimes."
        ),
        backstory=(
            "Carry specialist met diepgaand begrip van perpetual funding mechanics. "
            "Expert in cross-venue funding arbitrage en basis trading. "
            "Bekend om consistente carry extractie met gecontroleerd risico."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_carry_trader_i_agent() -> Agent:
    """Maak Agent 43: Carry Trader I (Funding Rate).

    Funding rate capture specialist.
    Gebruikt light LLM voor carry uitvoering.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Carry Trader I (Funding Rate) — Carry Trading",
        goal=(
            "Vang funding rate differentiëlen op. "
            "Monitor funding rates over grote perpetuals. "
            "Voer funding capture posities uit voor settlement. "
            "Volg funding betaling timing en omvang. "
            "Schaal posities op basis van funding regime persistentie."
        ),
        backstory=(
            "Funding rate specialist met timing expertise. "
            "Expert in het voorspellen van funding regime verschuivingen. "
            "Bekend om consistente funding capture met juiste timing."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_carry_trader_ii_agent() -> Agent:
    """Maak Agent 44: Carry Trader II (Basis/Calendar).

    Basis en calendar spread specialist.
    Gebruikt light LLM voor basis uitvoering.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Carry Trader II (Basis/Calendar) — Carry Trading",
        goal=(
            "Voer basis en calendar spread strategieën uit. "
            "Monitor spot-futures basis over instrumenten. "
            "Handel calendar spreads op kwartaal futures. "
            "Vang basis convergentie kansen op. "
            "Beheer roll risico en settlement timing."
        ),
        backstory=(
            "Basis trading specialist met calendar spread expertise. "
            "Expert in convergentie trades en roll dynamiek. "
            "Bekend om het extraheren van yield uit term structure."
        ),
        tools=tools,
        allow_delegation=False,
    )
