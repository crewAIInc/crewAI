"""STAFF-05: Chief Compliance & Legal Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import (
    AlertSystemTool,
)


def create_compliance_agent() -> Agent:
    """Maak STAFF-05 Chief Compliance & Legal Agent.

    Verantwoordelijk voor:
    - Beheer van beperkte lijst
    - Marktgedrag monitoring
    - Archivering en registratie
    - Compliance stops handhaving

    Rapporteert aan: STAFF-00 (CEO)
    Gebruikt lichte LLM voor compliance operaties.
    """
    return create_light_agent(
        role="Chief Compliance & Legal — Regelgeving en Marktgedrag",
        goal=(
            "Handhaaf beperkte lijst en marktgedragregels. Onderhoud registratie "
            "en compliance stops. Voer periodieke training en beleidscontroles uit. Beheer incident/"
            "case escalatie workflow. Zet geavanceerde tools in (chain analytics, AI surveillance) "
            "om overtredingen vroeg te detecteren zonder handel te vertragen."
        ),
        backstory=(
            "Senior compliance professional met diepe expertise in crypto-regelgeving, "
            "detectie van marktmanipulatie en handelstoezicht. Achtergrond in zowel "
            "traditionele finance compliance als crypto-native regelgevende frameworks. "
            "Bekend om het bouwen van compliance programma's die het bedrijf beschermen zonder "
            "een bottleneck te zijn voor handel. Expert in communicatiearchivering en registratie"
            "vereisten."
        ),
        tools=[
            AlertSystemTool(),
        ],
        allow_delegation=False,
    )
