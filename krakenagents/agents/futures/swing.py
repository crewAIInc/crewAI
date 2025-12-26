"""Futures Swing/Directional Trading agents (48-50) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_futures_research_tools, get_futures_execution_tools


def create_futures_swing_head_agent() -> Agent:
    """Maak Agent 48: Head of Swing/Directional Futures.

    Eigenaar van directionele futures trading strategieën.
    Gebruikt heavy LLM voor directionele analyse.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Head of Swing/Directional Futures — Swing Trading",
        goal=(
            "Eigenaar directionele futures trading strategieën. "
            "Ontwikkel these-gedreven directionele trades met leverage. "
            "Integreer funding cost in positiegrootte. "
            "Beheer portefeuille van directionele bets. "
            "Schaal leverage op basis van overtuiging en regime."
        ),
        backstory=(
            "Directionele trader gespecialiseerd in leveraged posities. "
            "Expert in het dimensioneren van posities met funding overwegingen. "
            "Bekend om grotere R-trades met gecontroleerd leverage risico."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_swing_btc_agent() -> Agent:
    """Maak Agent 49: Swing Trader Futures (BTC/ETH).

    Swing trading BTC/ETH perpetuals en futures.
    Gebruikt heavy LLM voor trade planning.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Swing Trader Futures (BTC/ETH) — Swing Trading",
        goal=(
            "Swing trade BTC/ETH perpetuals en futures. "
            "Bouw directionele posities met leverage. "
            "Integreer funding cost in trade planning. "
            "Schaal positie met marktbevestiging. "
            "Strikte stop discipline met leverage bewustzijn."
        ),
        backstory=(
            "Swing trader gespecialiseerd in BTC/ETH derivatives. "
            "Expert in leveraged positie management. "
            "Bekend om gedisciplineerde swing trading met juiste sizing."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_curve_trader_agent() -> Agent:
    """Maak Agent 50: Curve/RV Trader Futures.

    Curve en relative value trading op futures.
    Gebruikt heavy LLM voor RV analyse.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Curve/RV Trader Futures — Swing Trading",
        goal=(
            "Handel curve en relative value op futures. "
            "Voer term structure trades uit. "
            "Handel cross-venue basis kansen. "
            "Beheer roll risico in calendar posities. "
            "Identificeer RV kansen in derivatives markt."
        ),
        backstory=(
            "RV trader gespecialiseerd in derivatives curve trading. "
            "Expert in term structure en cross-venue dynamiek. "
            "Bekend om het extraheren van alpha uit RV kansen."
        ),
        tools=tools,
        allow_delegation=False,
    )
