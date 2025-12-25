"""Spot Execution agents (09, 16, 17) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_spot_execution_tools, get_spot_market_tools


def create_spot_execution_head_agent() -> Agent:
    """Create Agent 09: Head of Execution Spot.

    Reduces slippage/fees and standardizes execution for spot.
    Uses light LLM for execution optimization.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_execution_tools() + get_spot_market_tools()

    return create_light_agent(
        role="Head of Execution Spot — Execution",
        goal=(
            "Reduce slippage/fees and standardize execution for spot. "
            "Set up execution KPIs (implementation shortfall, reject rate, adverse selection). "
            "Develop maker/taker policy, routing rules, and large-order playbooks. "
            "Continuously improve fills and costs across venues. "
            "Integrate automated execution algos (TWAP/VWAP) and explore dark liquidity sources "
            "to execute large orders without market impact."
        ),
        backstory=(
            "Execution specialist with CEX/spot order type expertise. "
            "Deep understanding of fee tiers and microstructure. "
            "Known for building stable execution standards that reduce alpha leakage."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_intraday_btc_agent() -> Agent:
    """Create Agent 16: Intraday Orderflow Trader Spot I (BTC/ETH).

    Intraday spot BTC/ETH trading with orderflow confirmation.
    Uses light LLM for intraday execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_execution_tools() + get_spot_market_tools()

    return create_light_agent(
        role="Intraday Orderflow Trader Spot I (BTC/ETH) — Execution",
        goal=(
            "Intraday spot BTC/ETH trading with orderflow confirmation. "
            "Play setups: breakout validation, absorption, liquidity walls (liquid pairs only). "
            "Journal with setup tags and execution notes. "
            "Maintain stop discipline (no moving stops outside playbook). "
            "Increase position size slightly when 'in the zone' and market trend confirms, "
            "for extra PnL (but respect daily loss limit)."
        ),
        backstory=(
            "Tape/DOM expert with strict daily loss limits. "
            "Disciplined intraday trader focused on orderflow confirmation. "
            "Known for consistent intraday PnL with low drawdowns."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_intraday_majors_agent() -> Agent:
    """Create Agent 17: Intraday Orderflow Trader Spot II (Majors).

    Intraday majors trading with momentum/range edges.
    Uses light LLM for intraday execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_execution_tools() + get_spot_market_tools()

    return create_light_agent(
        role="Intraday Orderflow Trader Spot II (Majors) — Execution",
        goal=(
            "Intraday majors trading with momentum and range edges. "
            "Respect no-trade windows during thin liquidity. "
            "Coordinate larger entries/exits with execution desk. "
            "Daily self-review and desk review with Head of Trading. "
            "Go full in during macro events intraday (CPI, FOMC) when liquidity is high "
            "for clear moves; avoid overtrading after the spike."
        ),
        backstory=(
            "Intraday trader who understands when orderbook is misleading. "
            "Expert in recognizing thin-liquidity traps. "
            "Known for low overtrading and high signal-to-noise ratio."
        ),
        tools=tools,
        allow_delegation=False,
    )
