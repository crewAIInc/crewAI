"""Futures Execution agents (41, 51) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_futures_execution_tools, get_futures_market_tools


def create_futures_execution_head_agent() -> Agent:
    """Create Agent 41: Head of Execution Futures.

    Reduces slippage/fees and standardizes execution for futures.
    Uses heavy LLM for strategic execution decisions.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_execution_tools() + get_futures_market_tools()

    return create_heavy_agent(
        role="Head of Execution Futures — Execution",
        goal=(
            "Reduce slippage/fees and standardize execution for futures. "
            "Set up execution KPIs including funding impact. "
            "Develop maker/taker policy for derivatives venues. "
            "Build playbooks for large position entry/exit. "
            "Integrate algos for futures execution (TWAP/POV)."
        ),
        backstory=(
            "Execution specialist with derivatives expertise. "
            "Deep understanding of perpetual swap execution. "
            "Known for minimizing execution costs in leveraged trading."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_unwind_specialist_agent() -> Agent:
    """Create Agent 51: Position Unwind Specialist Futures.

    Specialist in unwinding large or distressed positions.
    Uses light LLM for unwind execution.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_futures_execution_tools() + get_futures_market_tools()

    return create_light_agent(
        role="Position Unwind Specialist Futures — Execution",
        goal=(
            "Specialize in unwinding large or distressed futures positions. "
            "Execute emergency position reductions. "
            "Minimize market impact during forced unwinds. "
            "Coordinate with risk on margin-driven reductions. "
            "24/7 availability for unwind situations."
        ),
        backstory=(
            "Unwind specialist with crisis execution experience. "
            "Expert in minimizing impact during forced liquidations. "
            "Known for calm execution under pressure."
        ),
        tools=tools,
        allow_delegation=False,
    )
