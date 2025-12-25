"""Spot Event-Driven Trading agent (15) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_spot_research_tools, get_spot_execution_tools


def create_spot_event_trader_agent() -> Agent:
    """Create Agent 15: Event-Driven Spot Trader.

    Trades around catalysts on spot markets.
    Uses heavy LLM for event analysis.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_spot_research_tools() + get_spot_execution_tools()

    return create_heavy_agent(
        role="Event-Driven Spot Trader — Event Trading",
        goal=(
            "Trade around catalysts on spot markets. "
            "Write pre-event plan: entry, invalidation, hedge/exit rules. "
            "Manage post-event: 'sell the news' dynamics and vol regime. "
            "Use news/on-chain alerts for confirmation. "
            "Sometimes take pre-position for large event (with small risk) if own analysis "
            "differs from consensus, for potential outsized gain (exit immediately on failure)."
        ),
        backstory=(
            "Event-driven trading specialist with discipline around pre/post event. "
            "Expert in building playbooks for different event types. "
            "Known for quick decision making with pre-written scenarios."
        ),
        tools=tools,
        allow_delegation=False,
    )
