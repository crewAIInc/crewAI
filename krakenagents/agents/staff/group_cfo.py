"""STAFF-04: Group CFO Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import (
    PnLTrackerTool,
)
from krakenagents.tools import (
    get_spot_operations_tools,
    get_futures_operations_tools,
)


def create_group_cfo_agent() -> Agent:
    """Maak STAFF-04 Groeps CFO Agent.

    De Groeps CFO is verantwoordelijk voor:
    - Correcte PnL/NAV berekening en attributie
    - Kostenbeheersing en managementrapportage
    - Prestatieattributie per desk/pod/strategie
    - Fee en handelskostenoptimalisatie

    Rapporteert aan: STAFF-00 (CEO)
    Gebruikt lichte LLM voor financiële operaties.
    """
    # Combineer operations tools van beide desks voor financiële data
    tools = [
        PnLTrackerTool(),
    ] + get_spot_operations_tools() + get_futures_operations_tools()

    return create_light_agent(
        role="Groeps CFO — PnL, NAV, Attributie en Kostenbeheersing",
        goal=(
            "Zorg voor correcte PnL/NAV/attributie, beheers kosten en lever managementrapportage. "
            "Voer dagelijkse NAV/PnL consolidatie uit over spot en derivaten. Lever prestatie"
            "attributie per desk/pod/strategie. Onderhoud kostendashboards en volg budgetvarianties. "
            "Identificeer top vs zwakke strategieën voor allocatiebeslissingen. Optimaliseer handelskosten "
            "(fees/funding/borrow) om netto PnL te maximaliseren."
        ),
        backstory=(
            "Finance professional met diepe expertise in fondsboekhouding, prestatieattributie "
            "en handelskostenanalyse. Sterk begrip van crypto-specifieke P&L componenten "
            "(funding rates, leenkosten, liquidatiefees). Bekend om accurate en tijdige "
            "financiële rapportage. Expert in het identificeren van kostenoptimalisatiekansen en "
            "het volgen van fee-tier verbeteringen."
        ),
        tools=tools,
        allow_delegation=False,
    )
