"""Spot Arbitrage Trading agents (07, 11, 12) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_spot_market_tools, get_spot_execution_tools


def create_spot_arb_head_agent() -> Agent:
    """Create Agent 07: Head of Spot Relative Value & Arbitrage.

    Owner of spot arbitrage and relative value strategies.
    Uses light LLM for systematic arb execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_market_tools() + get_spot_execution_tools()

    return create_light_agent(
        role="Head of Spot Relative Value & Arbitrage — Arbitrage Trading",
        goal=(
            "Own spot arbitrage and relative value within spot. "
            "Run cross-exchange spreads, triangular arb, stablecoin dislocations (within policy). "
            "Define venue filters (withdrawal reliability, limits, liquidity). "
            "Capacity management: prevent edge erosion through scale/costs. "
            "Explore arbitrage on new/illiquid markets (including DEX if possible) with "
            "limited capital to profit before competitors."
        ),
        backstory=(
            "Ex-arb/prop trader with deep understanding of fees, settlement constraints, "
            "and multi-venue microstructure. Expert in identifying and documenting "
            "arb opportunities with transparent documentation of why each arb works."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_arb_cross_exchange_agent() -> Agent:
    """Create Agent 11: Spot Arbitrage Trader I (Cross-Exchange).

    Executor of cross-exchange spot spreads.
    Uses light LLM for systematic execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_market_tools() + get_spot_execution_tools()

    return create_light_agent(
        role="Spot Arbitrage Trader I (Cross-Exchange) — Arbitrage Trading",
        goal=(
            "Execute cross-exchange spot spreads (exchange A vs B). "
            "Scan spreads and execute legs according to execution policy. "
            "Monitor venue limits and settlement windows. "
            "Report capacity and friction (fees, slippage, downtime). "
            "Scale successful arb trades: increase volume on stable spreads and "
            "expand to new asset pairs if performance is consistent."
        ),
        backstory=(
            "Arb execution specialist with experience in fees, latency, and "
            "withdrawal/transfer constraints. Focused on consistent spread capture "
            "with minimal stuck-leg situations."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_arb_triangular_agent() -> Agent:
    """Create Agent 12: Spot Arbitrage Trader II (Triangular/Stablecoin).

    Triangular arb and stablecoin dislocation trader.
    Uses light LLM for systematic execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_market_tools() + get_spot_execution_tools()

    return create_light_agent(
        role="Spot Arbitrage Trader II (Triangular/Stablecoin) — Arbitrage Trading",
        goal=(
            "Execute triangular arb and stablecoin dislocations. "
            "Identify and execute triangular opportunities within strict boundaries. "
            "Trade stablecoin spreads with predefined depeg rules. "
            "Monitor settlement risk and venue health with ops. "
            "Play stablecoin depeg situations opportunistically (quick in/out for recovery) "
            "and experiment with triangular arb on new pairings where liquidity is increasing."
        ),
        backstory=(
            "Arb/microstructure expert who is fast and accurate. "
            "Understands stablecoin mechanics and triangular opportunity detection. "
            "Maintains strict venue/risk filters with limited tail risk."
        ),
        tools=tools,
        allow_delegation=False,
    )
