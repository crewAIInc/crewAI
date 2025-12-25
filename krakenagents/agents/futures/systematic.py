"""Futures Systematic Trading agents (37, 42) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_futures_research_tools, get_futures_execution_tools


def create_futures_systematic_head_agent() -> Agent:
    """Create Agent 37: Head of Systematic Futures.

    Owner of systematic futures strategies.
    Uses heavy LLM for strategy design and analysis.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Head of Systematic Futures — Systematic Trading",
        goal=(
            "Own systematic futures strategies: funding rate models, basis strategies, momentum. "
            "Design and maintain signal library for derivatives. "
            "Write strategy specs including funding cost assumptions. "
            "Monthly model review with focus on funding regime changes. "
            "Develop AI/ML models for funding rate prediction and basis dynamics."
        ),
        backstory=(
            "Systematic/quant PM specialized in derivatives. "
            "Deep understanding of perpetual mechanics and funding cycles. "
            "Expert in incorporating funding costs into strategy returns."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_systematic_operator_agent() -> Agent:
    """Create Agent 42: Systematic Portfolio Operator Futures.

    Daily operator of live systematic futures strategies.
    Uses light LLM for operational execution.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_futures_execution_tools()

    return create_light_agent(
        role="Systematic Portfolio Operator Futures — Systematic Trading",
        goal=(
            "Daily operator of live systematic futures strategies. "
            "Run signals, monitor funding rates, execute rebalances. "
            "Track funding payments and their impact on positions. "
            "Pause strategy on anomalies per SOP. "
            "Provide feedback on execution friction and funding slippage."
        ),
        backstory=(
            "Systematic trader/operator with derivatives experience. "
            "Expert in running automated strategies with funding considerations. "
            "Quick at detecting funding regime changes."
        ),
        tools=tools,
        allow_delegation=False,
    )
