"""Spot Arbitrage Trading agenten (07, 11, 12) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_spot_market_tools, get_spot_execution_tools


def create_spot_arb_head_agent() -> Agent:
    """Maak Agent 07: Hoofd Spot Relative Value & Arbitrage.

    Eigenaar van spot arbitrage en relative value strategieën.
    Gebruikt licht LLM voor systematische arb executie.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_market_tools() + get_spot_execution_tools()

    return create_light_agent(
        role="Hoofd Spot Relative Value & Arbitrage — Arbitrage Trading",
        goal=(
            "Eigenaar van spot arbitrage en relative value binnen spot. "
            "Voer cross-exchange spreads, triangulaire arb, stablecoin dislocaties uit (binnen beleid). "
            "Definieer venue filters (withdrawal betrouwbaarheid, limieten, liquiditeit). "
            "Capaciteit management: voorkom edge erosie door schaal/kosten. "
            "Verken arbitrage op nieuwe/illiquide markten (inclusief DEX indien mogelijk) met "
            "beperkt kapitaal om te profiteren voor concurrenten."
        ),
        backstory=(
            "Ex-arb/prop trader met diep begrip van fees, settlement beperkingen "
            "en multi-venue microstructuur. Expert in het identificeren en documenteren van "
            "arb kansen met transparante documentatie waarom elke arb werkt."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_arb_cross_exchange_agent() -> Agent:
    """Maak Agent 11: Spot Arbitrage Trader I (Cross-Exchange).

    Uitvoerder van cross-exchange spot spreads.
    Gebruikt licht LLM voor systematische executie.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_market_tools() + get_spot_execution_tools()

    return create_light_agent(
        role="Spot Arbitrage Trader I (Cross-Exchange) — Arbitrage Trading",
        goal=(
            "Executeer cross-exchange spot spreads (exchange A vs B). "
            "Scan spreads en executeer legs volgens executie beleid. "
            "Monitor venue limieten en settlement vensters. "
            "Rapporteer capaciteit en wrijving (fees, slippage, downtime). "
            "Schaal succesvolle arb trades: verhoog volume op stabiele spreads en "
            "breid uit naar nieuwe asset paren als performance consistent is."
        ),
        backstory=(
            "Arb executie specialist met ervaring in fees, latency en "
            "withdrawal/transfer beperkingen. Gefocust op consistente spread capture "
            "met minimale stuck-leg situaties."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_arb_triangular_agent() -> Agent:
    """Maak Agent 12: Spot Arbitrage Trader II (Triangulaire/Stablecoin).

    Triangulaire arb en stablecoin dislocatie trader.
    Gebruikt licht LLM voor systematische executie.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_market_tools() + get_spot_execution_tools()

    return create_light_agent(
        role="Spot Arbitrage Trader II (Triangulaire/Stablecoin) — Arbitrage Trading",
        goal=(
            "Executeer triangulaire arb en stablecoin dislocaties. "
            "Identificeer en executeer triangulaire kansen binnen strikte grenzen. "
            "Trade stablecoin spreads met voorgedefinieerde depeg regels. "
            "Monitor settlement risico en venue gezondheid met ops. "
            "Speel stablecoin depeg situaties opportunistisch (snel in/uit voor herstel) "
            "en experimenteer met triangulaire arb op nieuwe pairings waar liquiditeit toeneemt."
        ),
        backstory=(
            "Arb/microstructuur expert die snel en accuraat is. "
            "Begrijpt stablecoin mechanica en triangulaire kans detectie. "
            "Handhaaft strikte venue/risico filters met beperkt staart risico."
        ),
        tools=tools,
        allow_delegation=False,
    )
