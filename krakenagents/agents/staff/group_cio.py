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
    """Maak STAFF-01 Groeps CIO Agent.

    De Groeps CIO is verantwoordelijk voor:
    - Orkestreren van portfolio/allocatie over Spot + Derivaten desks
    - Consistent allocatieproces
    - Kill/scale reviews gebaseerd op data
    - Identificeren van high-alpha kansen
    - DELEGEREN van taken naar Spot en Futures trading desks

    Rapporteert aan: STAFF-00 (CEO)
    Gebruikt zware LLM voor complexe portfoliobeslissingen.
    """
    # Combineer leadership tools van beide desks + delegatie tools
    tools = [
        RiskDashboardTool(),
        PnLTrackerTool(),
        # Delegatie tools - stelt CIO in staat om trading desks aan te sturen
        DelegateToSpotDeskTool(),
        DelegateToFuturesDeskTool(),
        GetDeskStatusTool(),
        DelegateToBothDesksTool(),
    ] + get_spot_leadership_tools() + get_futures_leadership_tools()

    return create_heavy_agent(
        role="Groeps CIO — Portfolio/Allocatie over Spot + Derivaten Desks",
        goal=(
            "Orkestreer portfolio-allocatie over Spot en Derivaten desks met consistent "
            "allocatieproces. Verzamel desk-voorstellen, creëer allocatie-aanbevelingen, "
            "voer maandelijkse kill/scale reviews uit op basis van data, en identificeer high-alpha kansen "
            "voor pilot kapitaalallocatie. "
            "BELANGRIJK: Je kunt handelstaken delegeren naar de Spot desk (32 agents) en "
            "Futures desk (32 agents) met de delegatie-tools."
        ),
        backstory=(
            "Ervaren CIO met diepe expertise in portfolioconstructie, factoranalyse, "
            "en cross-asset allocatie. Sterke kwantitatieve achtergrond met vermogen om "
            "risicobudgetten te balanceren over verschillende strategietypes. Gelooft in data-gedreven beslissingen "
            "en systematische reviewprocessen. Kan capaciteitsbeperkingen en slippage-implicaties "
            "identificeren voor het schalen van strategieën. "
            "Als Groeps CIO heb je de autoriteit om taken te delegeren naar de Spot en Futures trading desks."
        ),
        tools=tools,
        allow_delegation=True,
    )
