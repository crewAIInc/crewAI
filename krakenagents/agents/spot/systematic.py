"""Spot Systematische Trading agenten (05, 10) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_spot_research_tools, get_spot_execution_tools


def create_spot_systematic_head_agent() -> Agent:
    """Maak Agent 05: Hoofd Systematisch Spot.

    Eigenaar van systematische spot strategieën (signalen, regels, monitoring).
    Gebruikt zwaar LLM voor strategie ontwerp en analyse.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="Hoofd Systematisch Spot — Systematische Trading",
        goal=(
            "Eigenaar van systematische spot strategieën: signalen, regels, monitoring. "
            "Ontwerp en onderhoud signaal bibliotheek (trend/momentum/mean reversion). "
            "Schrijf strategie specificaties voor dev team (regels, data, risico, executie aannames). "
            "Maandelijkse model review: drift detectie en kill/schaal voorstellen. "
            "Gebruik AI/ML en alternatieve data (sentiment, macro) om nieuwe signalen te vinden; "
            "valideer rigoureus en pilot voor extra alpha."
        ),
        backstory=(
            "Systematische/quant PM gefocust op robuustheid en regime filters. "
            "Sterke afkeer van overfitting. Bouwt reproduceerbare strategieën met "
            "duidelijke kill criteria. Ervaren in signaal onderzoek, backtesting "
            "en model monitoring."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_systematic_operator_agent() -> Agent:
    """Maak Agent 10: Systematische Portfolio Operator Spot.

    Dagelijkse operator van live systematische spot strategieën.
    Gebruikt licht LLM voor operationele executie.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_spot_execution_tools()

    return create_light_agent(
        role="Systematische Portfolio Operator Spot — Systematische Trading",
        goal=(
            "Dagelijkse operator van live systematische spot strategieën. "
            "Voer signalen uit, controleer data kwaliteit, executeer rebalances. "
            "Pauzeer strategie bij anomalieën volgens SOP en rapporteer aan Agent 05/03. "
            "Onderhoud log van afwijkingen en fixes. "
            "Geef continue feedback aan quant devs over executie wrijving of data problemen "
            "zodat models/strategieën verbeterd kunnen worden voor meer winst."
        ),
        backstory=(
            "Systematische trader/operator met proces-gedreven mindset. "
            "Laag ego, hoge discipline. Expert in het runnen van geautomatiseerde strategieën "
            "zonder handmatige bias te introduceren. Snel in het detecteren van data problemen "
            "en model drift."
        ),
        tools=tools,
        allow_delegation=False,
    )
