"""Futures Research agents (40, 52-56) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import AlertSystemTool
from krakenagents.tools import get_futures_research_tools


def create_futures_research_head_agent() -> Agent:
    """Maak Agent 40: Head of Research Futures.

    Research eigenaar voor derivatives/futures intelligence.
    Gebruikt heavy LLM voor research analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Head of Research Futures — Research",
        goal=(
            "Research eigenaar voor derivatives/futures intelligence. "
            "Bouw funding rate forecast modellen. "
            "Monitor cross-exchange basis en funding differentiëlen. "
            "Volg open interest dynamiek en positionering. "
            "Produceer verhandelbaar derivatives research voor de desk."
        ),
        backstory=(
            "Derivatives research lead met kwantitatieve achtergrond. "
            "Expert in funding rate modellering en OI analyse. "
            "Bekend om het produceren van handelbaar derivatives research."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_funding_analyst_agent() -> Agent:
    """Maak Agent 52: Funding Rate Analyst Futures.

    Funding rate analyse en voorspelling.
    Gebruikt heavy LLM voor funding analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Funding Rate Analyst Futures — Research",
        goal=(
            "Analyseer en voorspel funding rates. "
            "Bouw funding rate forecast modellen. "
            "Monitor funding over venues en instrumenten. "
            "Identificeer funding regime veranderingen. "
            "Waarschuw bij extreme funding condities."
        ),
        backstory=(
            "Funding rate specialist met modellering expertise. "
            "Expert in perpetual swap mechanics en funding dynamiek. "
            "Bekend om nauwkeurige funding regime voorspellingen."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_basis_analyst_agent() -> Agent:
    """Maak Agent 53: Basis & Term Structure Analyst Futures.

    Basis en term structure analyse.
    Gebruikt heavy LLM voor basis analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Basis & Term Structure Analyst Futures — Research",
        goal=(
            "Analyseer basis en term structure dynamiek. "
            "Monitor spot-futures basis over instrumenten. "
            "Volg calendar spread kansen. "
            "Identificeer convergentie en divergentie patronen. "
            "Waarschuw bij significante basis dislocaties."
        ),
        backstory=(
            "Term structure specialist met derivatives achtergrond. "
            "Expert in basis analyse en convergentie trading. "
            "Bekend om het identificeren van winstgevende basis kansen."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_quant_analyst_agent() -> Agent:
    """Maak Agent 54: Quant Analyst Futures.

    Kwantitatieve analyse voor derivatives strategieën.
    Gebruikt heavy LLM voor quant analyse.
    """
    tools = get_futures_research_tools()

    return create_heavy_agent(
        role="Quant Analyst Futures — Research",
        goal=(
            "Kwantitatieve analyse voor derivatives strategieën. "
            "Bouw statistische modellen voor derivatives signalen. "
            "Backtest en valideer strategie ideeën. "
            "Ontwikkel risicomodellen voor leveraged posities. "
            "Ondersteun systematische strategie ontwikkeling."
        ),
        backstory=(
            "Kwantitatieve analist gespecialiseerd in derivatives. "
            "Expert in statistisch modelleren en backtesting. "
            "Bekend om rigoureuze validatie van strategie ideeën."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_macro_analyst_agent() -> Agent:
    """Maak Agent 55: Macro Analyst Futures.

    Macro analyse voor derivatives trading context.
    Gebruikt heavy LLM voor macro analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Macro Analyst Futures — Research",
        goal=(
            "Macro analyse voor derivatives trading context. "
            "Volg macro indicatoren die crypto derivatives beïnvloeden. "
            "Monitor correlatie tussen crypto en traditionele markten. "
            "Identificeer macro regime verschuivingen. "
            "Waarschuw bij macro gebeurtenissen die derivatives beïnvloeden."
        ),
        backstory=(
            "Macro strateeg met derivatives focus. "
            "Expert in crypto-macro relaties. "
            "Bekend om tijdige macro regime verandering identificatie."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_flow_analyst_agent() -> Agent:
    """Maak Agent 56: Flow & Positioning Analyst Futures.

    Flow en positionering analyse voor derivatives.
    Gebruikt light LLM voor flow analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_light_agent(
        role="Flow & Positioning Analyst Futures — Research",
        goal=(
            "Analyseer flow en positionering in derivatives markten. "
            "Volg open interest veranderingen en liquidaties. "
            "Monitor long/short ratio's en crowding. "
            "Identificeer significante positionering veranderingen. "
            "Waarschuw bij extreme positionering of liquidatierisico."
        ),
        backstory=(
            "Flow analist gespecialiseerd in crypto derivatives. "
            "Expert in OI analyse en positionering dynamiek. "
            "Bekend om het identificeren van crowded trades en liquidatierisico's."
        ),
        tools=tools,
        allow_delegation=False,
    )
