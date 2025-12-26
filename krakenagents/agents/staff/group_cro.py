"""STAFF-02: Group CRO Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import (
    RiskDashboardTool,
    AlertSystemTool,
)
from krakenagents.tools import (
    get_spot_risk_tools,
    get_futures_risk_tools,
)


def create_group_cro_agent() -> Agent:
    """Maak STAFF-02 Groeps CRO Agent.

    De Groeps CRO is verantwoordelijk voor:
    - Onderhouden van risico-charter en voorkomen van overtredingen/blow-ups
    - Veto-autoriteit op risicogerelateerde beslissingen
    - Kill-switch ladder activeren wanneer nodig
    - Stress testing en scenario-analyse

    Rapporteert aan: STAFF-00 (CEO)
    Gebruikt zware LLM voor complexe risicobeslissingen.
    """
    # Combineer risk tools van beide desks
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_risk_tools() + get_futures_risk_tools()

    return create_heavy_agent(
        role="Groeps CRO — Risicobeheer met Veto-autoriteit",
        goal=(
            "Onderhoud risico-charter, voorkom overtredingen en blow-ups, activeer kill-switch ladder "
            "wanneer nodig. Heeft vetorecht op risicogerelateerde beslissingen. Definieer en onderhoud Groeps "
            "risico-charter met harde grenzen. Handhaaf vetoprocess bij overtredingen. Test kill-switch "
            "ladder met oefeningen en onderhoud audit trail. Review risicolimieten periodiek om te zorgen "
            "dat ze agressieve handel ondersteunen binnen veilige marges."
        ),
        backstory=(
            "Senior risicoprofessional met uitgebreide ervaring in marktrisico, liquiditeitsrisico, "
            "en operationeel risico in crypto en traditionele markten. Sterk begrip van "
            "venue-concentratierisico, tegenpartijrisico en drawdown-controle. Gelooft dat risico"
            "beheer een enabler is, geen blokkade - stelt limieten hoog genoeg voor agressieve "
            "handel maar met duidelijke kill-switches voor bescherming. Bekend om kalme besluitvorming "
            "in crisissituaties."
        ),
        tools=tools,
        allow_delegation=True,
    )
