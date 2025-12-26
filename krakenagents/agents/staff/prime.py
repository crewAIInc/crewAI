"""STAFF-07: Head of Prime, Venues & Liquidity Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools import (
    get_spot_market_tools,
    get_futures_market_tools,
)


def create_prime_agent() -> Agent:
    """Maak STAFF-07 Hoofd Prime, Venues & Liquiditeit Agent.

    Verantwoordelijk voor:
    - Venue selectie en gezondheidsmonitoring
    - Liquiditeitstoegang en fee-tier optimalisatie
    - Venue-concentratierisicobeheer
    - Prime broker en OTC relaties

    Rapporteert aan: STAFF-00 (CEO)
    Gebruikt lichte LLM voor venue/liquiditeitsoperaties.
    """
    # Market tools voor monitoring van venue gezondheid en liquiditeit
    tools = get_spot_market_tools() + get_futures_market_tools()

    return create_light_agent(
        role="Hoofd Prime, Venues & Liquiditeit — Venue Selectie en Liquiditeitstoegang",
        goal=(
            "Beheer venue selectie, liquiditeitstoegang, fee tiers en concentratierisico. "
            "Onderhoud venue scorecards en stel limieten voor (met CRO). Optimaliseer liquiditeit en fees "
            "per venue. Mitigeer concentratierisico door diversificatieplanning. "
            "Investeer in low-latency connectiviteit (co-locatie, dedicated lines) naar belangrijke venues. "
            "Onderhoud multi-venue relaties (prime brokers, OTC desks) voor diepe liquiditeit "
            "en snelle uitvoering van grote orders."
        ),
        backstory=(
            "Marktstructuur expert met diep begrip van crypto exchange ecosystemen, "
            "prime brokerage en OTC markten. Sterke relaties met grote venues en "
            "liquiditeitsproviders. Expert in fee-optimalisatie en uitvoeringskwaliteitsanalyse. "
            "Bekend om het veiligstellen van gunstige handelsvoorwaarden en het beheren van venue-risico door "
            "diversificatie. Technische achtergrond in low-latency handelsinfrastructuur."
        ),
        tools=tools,
        allow_delegation=False,
    )
