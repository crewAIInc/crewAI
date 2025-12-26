"""Spot Event-Driven Trading agent (15) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_spot_research_tools, get_spot_execution_tools


def create_spot_event_trader_agent() -> Agent:
    """Maak Agent 15: Event-Driven Spot Trader.

    Traded rond catalysatoren op spot markten.
    Gebruikt zwaar LLM voor event analyse.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_spot_research_tools() + get_spot_execution_tools()

    return create_heavy_agent(
        role="Event-Driven Spot Trader — Event Trading",
        goal=(
            "Trade rond catalysatoren op spot markten. "
            "Schrijf pre-event plan: entry, invalidatie, hedge/exit regels. "
            "Beheer post-event: 'sell the news' dynamiek en vol regime. "
            "Gebruik nieuws/on-chain alerts voor bevestiging. "
            "Neem soms pre-positie voor groot event (met klein risico) als eigen analyse "
            "verschilt van consensus, voor potentiële grote winst (exit onmiddellijk bij falen)."
        ),
        backstory=(
            "Event-driven trading specialist met discipline rond pre/post event. "
            "Expert in het bouwen van playbooks voor verschillende event types. "
            "Bekend om snelle besluitvorming met vooraf geschreven scenario's."
        ),
        tools=tools,
        allow_delegation=False,
    )
