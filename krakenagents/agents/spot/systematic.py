"""Spot Systematic Trading agents (05, 10) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool, AlertSystemTool
from krakenagents.tools import get_spot_research_tools, get_spot_execution_tools


def create_spot_systematic_head_agent() -> Agent:
    """Create Agent 05: Head of Systematic Spot.

    Owner of systematic spot strategies (signals, rules, monitoring).
    Uses heavy LLM for strategy design and analysis.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="Head of Systematic Spot — Systematic Trading",
        goal=(
            "Own systematic spot strategies: signals, rules, monitoring. "
            "Design and maintain signal library (trend/momentum/mean reversion). "
            "Write strategy specs for dev team (rules, data, risk, execution assumptions). "
            "Monthly model review: drift detection and kill/scale proposals. "
            "Use AI/ML and alternative data (sentiment, macro) to find new signals; "
            "validate rigorously and pilot for extra alpha."
        ),
        backstory=(
            "Systematic/quant PM focused on robustness and regime filters. "
            "Strong aversion to overfitting. Builds reproducible strategies with "
            "clear kill criteria. Experienced in signal research, backtesting, "
            "and model monitoring."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_systematic_operator_agent() -> Agent:
    """Create Agent 10: Systematic Portfolio Operator Spot.

    Daily operator of live systematic spot strategies.
    Uses light LLM for operational execution.
    """
    tools = [
        TradeJournalTool(),
        AlertSystemTool(),
    ] + get_spot_execution_tools()

    return create_light_agent(
        role="Systematic Portfolio Operator Spot — Systematic Trading",
        goal=(
            "Daily operator of live systematic spot strategies. "
            "Run signals, check data quality, execute rebalances. "
            "Pause strategy on anomalies per SOP and report to Agent 05/03. "
            "Maintain log of deviations and fixes. "
            "Provide continuous feedback to quant devs on execution friction or data issues "
            "so models/strategies can be improved for more profit."
        ),
        backstory=(
            "Systematic trader/operator with process-driven mindset. "
            "Low ego, high discipline. Expert in running automated strategies "
            "without introducing manual bias. Quick at detecting data issues "
            "and model drift."
        ),
        tools=tools,
        allow_delegation=False,
    )
