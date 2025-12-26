"""Spot Risicomanagement agenten (19, 25-27) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import RiskDashboardTool, AlertSystemTool
from krakenagents.tools import get_spot_risk_tools


def create_spot_inventory_coordinator_agent() -> Agent:
    """Maak Agent 19: Voorraad & Risico Coördinator Spot.

    Coördinator tussen traders en risico voor spot voorraad/exposure.
    Gebruikt licht LLM voor coördinatie taken.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_risk_tools()

    return create_light_agent(
        role="Voorraad & Risico Coördinator Spot — Risicomanagement",
        goal=(
            "Coördineer tussen traders en risico voor spot voorraad/exposure. "
            "Dagelijkse voorraad controles: concentraties, liquiditeit niveaus, exit gereedheid. "
            "Start inter-desk hedge verzoek als hedge alleen mogelijk via futures "
            "(zonder futures direct te traden). "
            "Signaleer mismatch tussen exposure en regime. "
            "Hedge niet te vroeg: laat beperkte overexposure lopen als markt gunstig is; "
            "hedge alleen wanneer risico asymmetrie toeneemt, voor betere risico/reward."
        ),
        backstory=(
            "Risico-bewuste trader/analist gefocust op concentratie en liquiditeit. "
            "Expert in coördineren tussen trading en risico functies. "
            "Bekend om het verminderen van onbedoeld beta/concentratie risico."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_risk_monitor_agent() -> Agent:
    """Maak Agent 25: Real-Time Risico Monitor Spot.

    24/7 monitoring van spot risico metrics en drempelwaarden.
    Gebruikt licht LLM voor monitoring taken.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_risk_tools()

    return create_light_agent(
        role="Real-Time Risico Monitor Spot — Risicomanagement",
        goal=(
            "24/7 monitoring van spot risico metrics en drempelwaarden. "
            "Monitor positie limieten, exposure caps en drawdown niveaus. "
            "Alert onmiddellijk bij drempelwaarde overschrijdingen. "
            "Escaleer naar CRO Spot bij kritieke alerts. "
            "Volg venue gezondheid en liquiditeit condities."
        ),
        backstory=(
            "Risico monitoring specialist met aandacht voor detail. "
            "Expert in real-time risico surveillance en alert management. "
            "Bekend om snelle escalatie bij kritieke problemen."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_limits_officer_agent() -> Agent:
    """Maak Agent 26: Limieten & Controles Officer Spot.

    Onderhoudt en handhaaft trading limieten en controles.
    Gebruikt licht LLM voor limieten management.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_risk_tools()

    return create_light_agent(
        role="Limieten & Controles Officer Spot — Risicomanagement",
        goal=(
            "Onderhoud en handhaaf trading limieten en controles voor spot. "
            "Documenteer en update limiet frameworks. "
            "Verwerk limiet wijzigingsverzoeken met juiste goedkeuringen. "
            "Verzeker limiet handhaving in trading systemen. "
            "Regelmatige limiet reviews en kalibratie."
        ),
        backstory=(
            "Controles specialist met trading limieten expertise. "
            "Expert in limiet frameworks en governance. "
            "Bekend om het handhaven van consistente limiet handhaving."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_margin_analyst_agent() -> Agent:
    """Maak Agent 27: Margin & Collateral Analist Spot.

    Monitort margin en collateral voor spot posities.
    Gebruikt licht LLM voor margin analyse.
    """
    tools = [
        RiskDashboardTool(),
    ] + get_spot_risk_tools()

    return create_light_agent(
        role="Margin & Collateral Analist Spot — Risicomanagement",
        goal=(
            "Monitor margin en collateral voor spot posities. "
            "Volg margin gebruik over venues. "
            "Optimaliseer collateral allocatie. "
            "Alert bij margin calls of lage buffer situaties. "
            "Coördineer met treasury over collateral bewegingen."
        ),
        backstory=(
            "Margin specialist met exchange margin model expertise. "
            "Expert in collateral optimalisatie en margin efficiëntie. "
            "Bekend om het voorkomen van margin-gerelateerde trading verstoringen."
        ),
        tools=tools,
        allow_delegation=False,
    )
