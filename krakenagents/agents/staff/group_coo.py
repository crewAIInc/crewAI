"""STAFF-03: Group COO Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import (
    AlertSystemTool,
)
from krakenagents.tools import (
    get_spot_operations_tools,
    get_futures_operations_tools,
)


def create_group_coo_agent() -> Agent:
    """Maak STAFF-03 Groeps COO Agent.

    De Groeps COO is verantwoordelijk voor:
    - Run-the-business operaties en controles
    - Reconciliaties en settlement
    - Incident response coördinatie
    - Scheiding van taken handhaving

    Rapporteert aan: STAFF-00 (CEO)
    Gebruikt lichte LLM voor operationele taken.
    """
    # Combineer operations tools van beide desks
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools() + get_futures_operations_tools()

    return create_light_agent(
        role="Groeps COO — Operaties en Controles",
        goal=(
            "Run-the-business: beheer controles, reconciliaties en incident response. "
            "Handhaaf scheiding van Spot/Derivaten operationeel (accounts, goedkeuringen, rapportage). "
            "Voer dagelijkse ops control cyclus uit (breaks dezelfde dag opgelost). Coördineer incident response "
            "voor venue-uitval, settlementproblemen en ops-problemen. Schaal ops-processen voor "
            "hoge-volume periodes om reconciliaties en settlements vlekkeloos te laten verlopen."
        ),
        backstory=(
            "Ervaren operationeel leider met sterke achtergrond in handelsoperaties, "
            "middle office en settlement. Expert in procesontwerp, controleframeworks "
            "en incidentbeheer. Bekend om het bouwen van schaalbare operaties die piek"
            "volatiliteit aankunnen zonder vertragingen. Sterke focus op scheiding van taken en "
            "operationele risicomitigatie."
        ),
        tools=tools,
        allow_delegation=False,
    )
