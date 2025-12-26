"""Futures Systematic Trading agents (37, 42) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_futures_research_tools, get_futures_execution_tools


def create_futures_systematic_head_agent() -> Agent:
    """Maak Agent 37: Head of Systematic Futures.

    Eigenaar van systematische futures strategieën.
    Gebruikt heavy LLM voor strategie ontwerp en analyse.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Head of Systematic Futures — Systematic Trading",
        goal=(
            "Eigenaar systematische futures strategieën: funding rate modellen, basis strategieën, momentum. "
            "Ontwerp en onderhoud signaalbibliotheek voor derivatives. "
            "Schrijf strategie specificaties inclusief funding cost aannames. "
            "Maandelijkse model review met focus op funding regime veranderingen. "
            "Ontwikkel AI/ML modellen voor funding rate voorspelling en basis dynamiek."
        ),
        backstory=(
            "Systematische/quant PM gespecialiseerd in derivatives. "
            "Diepgaand begrip van perpetual mechanics en funding cycli. "
            "Expert in het integreren van funding costs in strategie rendementen."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_systematic_operator_agent() -> Agent:
    """Maak Agent 42: Systematic Portfolio Operator Futures.

    Dagelijkse operator van live systematische futures strategieën.
    Gebruikt light LLM voor operationele uitvoering.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_futures_execution_tools()

    return create_light_agent(
        role="Systematic Portfolio Operator Futures — Systematic Trading",
        goal=(
            "Dagelijkse operator van live systematische futures strategieën. "
            "Voer signalen uit, monitor funding rates, voer herbalanceringen uit. "
            "Volg funding betalingen en hun impact op posities. "
            "Pauzeer strategie bij anomalieën volgens SOP. "
            "Geef feedback over uitvoeringswrijving en funding slippage."
        ),
        backstory=(
            "Systematische trader/operator met derivatives ervaring. "
            "Expert in het draaien van geautomatiseerde strategieën met funding overwegingen. "
            "Snel in het detecteren van funding regime veranderingen."
        ),
        tools=tools,
        allow_delegation=False,
    )
