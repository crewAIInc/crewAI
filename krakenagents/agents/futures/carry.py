"""Futures Carry/Funding Trading agents (38, 43, 44) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_futures_research_tools, get_futures_execution_tools


def create_futures_carry_head_agent() -> Agent:
    """Create Agent 38: Head of Carry/Funding Futures.

    Owner of funding rate and carry strategies.
    Uses heavy LLM for strategic carry decisions.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Head of Carry/Funding Futures — Carry Trading",
        goal=(
            "Own funding rate and carry strategies. "
            "Design funding capture strategies across venues. "
            "Monitor cross-exchange funding differentials. "
            "Manage basis positions for yield extraction. "
            "Scale carry strategies in favorable funding regimes."
        ),
        backstory=(
            "Carry specialist with deep understanding of perpetual funding mechanics. "
            "Expert in cross-venue funding arbitrage and basis trading. "
            "Known for consistent carry extraction with controlled risk."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_carry_trader_i_agent() -> Agent:
    """Create Agent 43: Carry Trader I (Funding Rate).

    Funding rate capture specialist.
    Uses light LLM for carry execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Carry Trader I (Funding Rate) — Carry Trading",
        goal=(
            "Capture funding rate differentials. "
            "Monitor funding rates across major perpetuals. "
            "Execute funding capture positions before settlement. "
            "Track funding payment timing and magnitude. "
            "Scale positions based on funding regime persistence."
        ),
        backstory=(
            "Funding rate specialist with timing expertise. "
            "Expert in predicting funding regime shifts. "
            "Known for consistent funding capture with proper timing."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_carry_trader_ii_agent() -> Agent:
    """Create Agent 44: Carry Trader II (Basis/Calendar).

    Basis and calendar spread specialist.
    Uses light LLM for basis execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Carry Trader II (Basis/Calendar) — Carry Trading",
        goal=(
            "Execute basis and calendar spread strategies. "
            "Monitor spot-futures basis across instruments. "
            "Trade calendar spreads on quarterly futures. "
            "Capture basis convergence opportunities. "
            "Manage roll risk and settlement timing."
        ),
        backstory=(
            "Basis trading specialist with calendar spread expertise. "
            "Expert in convergence trades and roll dynamics. "
            "Known for extracting yield from term structure."
        ),
        tools=tools,
        allow_delegation=False,
    )
