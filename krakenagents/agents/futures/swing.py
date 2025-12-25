"""Futures Swing/Directional Trading agents (48-50) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_futures_research_tools, get_futures_execution_tools


def create_futures_swing_head_agent() -> Agent:
    """Create Agent 48: Head of Swing/Directional Futures.

    Owner of directional futures trading strategies.
    Uses heavy LLM for directional analysis.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Head of Swing/Directional Futures — Swing Trading",
        goal=(
            "Own directional futures trading strategies. "
            "Develop thesis-driven directional trades using leverage. "
            "Integrate funding cost into position sizing. "
            "Manage portfolio of directional bets. "
            "Scale leverage based on conviction and regime."
        ),
        backstory=(
            "Directional trader specialized in leveraged positions. "
            "Expert in sizing positions with funding considerations. "
            "Known for larger R-trades with controlled leverage risk."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_swing_btc_agent() -> Agent:
    """Create Agent 49: Swing Trader Futures (BTC/ETH).

    Swing trading BTC/ETH perpetuals and futures.
    Uses heavy LLM for trade planning.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Swing Trader Futures (BTC/ETH) — Swing Trading",
        goal=(
            "Swing trade BTC/ETH perpetuals and futures. "
            "Build directional positions with leverage. "
            "Incorporate funding cost into trade planning. "
            "Scale position with market confirmation. "
            "Strict stop discipline with leverage awareness."
        ),
        backstory=(
            "Swing trader specialized in BTC/ETH derivatives. "
            "Expert in leveraged position management. "
            "Known for disciplined swing trading with proper sizing."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_curve_trader_agent() -> Agent:
    """Create Agent 50: Curve/RV Trader Futures.

    Curve and relative value trading on futures.
    Uses heavy LLM for RV analysis.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Curve/RV Trader Futures — Swing Trading",
        goal=(
            "Trade curve and relative value on futures. "
            "Execute term structure trades. "
            "Trade cross-venue basis opportunities. "
            "Manage roll risk in calendar positions. "
            "Identify RV opportunities in derivatives market."
        ),
        backstory=(
            "RV trader specialized in derivatives curve trading. "
            "Expert in term structure and cross-venue dynamics. "
            "Known for extracting alpha from RV opportunities."
        ),
        tools=tools,
        allow_delegation=False,
    )
