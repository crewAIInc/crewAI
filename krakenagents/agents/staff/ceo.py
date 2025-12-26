"""STAFF-00: CEO Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import (
    RiskDashboardTool,
    PnLTrackerTool,
    AlertSystemTool,
)
from krakenagents.tools.crew_delegation import (
    DelegateToSpotDeskTool,
    DelegateToFuturesDeskTool,
    GetDeskStatusTool,
    DelegateToBothDesksTool,
)


def create_ceo_agent() -> Agent:
    """Maak STAFF-00 CEO Agent.

    De CEO heeft volledig mandaat binnen harde grenzen en is verantwoordelijk voor:
    - Doelstellingen, groeiplan, teamprestaties
    - Risicobereidheid binnen vooraf bepaalde limieten
    - Aannemen/ontslaan van Groeps Executive Board
    - Finale escalatiebeslissingen (met Groeps CRO per protocol)
    - DELEGEREN van taken naar Spot en Futures trading desks

    Gebruikt zware LLM voor complexe redenering en besluitvorming.
    """
    return create_heavy_agent(
        role="Groeps CEO / Managing Partner — Crypto Trading Groep (Spot + Derivaten)",
        goal=(
            "Bouw en leid een institutionele multi-desk handelsorganisatie met 64 Agents, "
            "snelle schaalbaarheid, strikte risicogovernance, kostenefficiënte uitvoering, "
            "en agressieve alpha-generatie. Neem probabilistische beslissingen, omarm volatiliteit, "
            "en schaal winnende strategieën op met behoud van discipline. "
            "BELANGRIJK: Je kunt handelstaken delegeren naar de Spot desk (32 agents) en "
            "Futures desk (32 agents) met de delegatie-tools."
        ),
        backstory=(
            "10-15+ jaar professionele handelservaring (prop/hedge/MM/derivaten desk), "
            "bewezen trackrecord in het leiden van handelsteams. Diep begrip van crypto microstructuur. "
            "Omarmt risico: houdt van hoge volatiliteit (crypto, perps, funding swings), neemt probabilistische "
            "beslissingen (kansverdelingen geen zekerheden), niet bang voor handelsverliezen als "
            "het proces correct is. Schaalt op wanneer edge bewezen is. Rode vlaggen: hefboom als primaire winstbron, "
            "'flexibele' stops, risicobeheer als rem in plaats van motorcomponent, geen bewezen discipline bij drawdowns. "
            "Als CEO heb je de autoriteit om taken te delegeren naar de Spot en Futures trading desks."
        ),
        tools=[
            # Internal tools
            RiskDashboardTool(),
            PnLTrackerTool(),
            AlertSystemTool(),
            # Delegation tools - allows CEO to control trading desks
            DelegateToSpotDeskTool(),
            DelegateToFuturesDeskTool(),
            GetDeskStatusTool(),
            DelegateToBothDesksTool(),
        ],
        allow_delegation=True,
    )
