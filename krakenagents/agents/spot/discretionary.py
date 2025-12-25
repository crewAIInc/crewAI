"""Spot Discretionary Trading agents (06, 13, 14) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_spot_research_tools, get_spot_execution_tools


def create_spot_discretionary_head_agent() -> Agent:
    """Create Agent 06: Head of Discretionary Spot (Themes & Swing).

    Owner of discretionary spot swing/thematic book.
    Uses heavy LLM for thesis-driven decisions.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_research_tools() + get_spot_execution_tools()

    return create_heavy_agent(
        role="Head of Discretionary Spot (Themes & Swing) — Discretionary Trading",
        goal=(
            "Own discretionary spot swing/thematic book. "
            "Build thesis-driven trades (days-weeks) with strict invalidation. "
            "Integrate research: tokenomics/unlocks, flows, catalysts. "
            "Manage positions: scale in/out, trailing, profit protection. "
            "Scout niche tokens/narratives early and take small positions (conscious high risk) "
            "for potential outsized gains if thesis plays out."
        ),
        backstory=(
            "Experienced swing trader with deep market structure knowledge. "
            "Expert in risk/reward analysis and theme rotation. Known for "
            "building larger R-trades with controlled drawdowns and clear exits."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_swing_majors_agent() -> Agent:
    """Create Agent 13: Discretionary Swing Trader Spot I (Majors).

    Swing trader for BTC/ETH and top-liquid coins.
    Uses heavy LLM for trade planning.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_research_tools() + get_spot_execution_tools()

    return create_heavy_agent(
        role="Discretionary Swing Trader Spot I (Majors) — Discretionary Trading",
        goal=(
            "Swing trade BTC/ETH and top-liquid coins. "
            "Plan trend continuation/pullback trades with invalidation levels. "
            "Combine levels with flows/volume (no indicator blindness). "
            "Build scenario trade plans (base/bull/bear). "
            "Let winners run: increase position or widen trailing stop when trade is "
            "convincingly winning to maximize trend capture (maintain stop discipline)."
        ),
        backstory=(
            "Multi-timeframe trader expert in market structure and risk/reward. "
            "Uses flow and volume analysis alongside technical levels. "
            "Builds repeatable playbooks with consistent execution."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_swing_alts_agent() -> Agent:
    """Create Agent 14: Discretionary Swing Trader Spot II (Alts/Themes).

    Swing/thematic trader in liquid alts within universe.
    Uses heavy LLM for theme analysis.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_research_tools() + get_spot_execution_tools()

    return create_heavy_agent(
        role="Discretionary Swing Trader Spot II (Alts/Themes) — Discretionary Trading",
        goal=(
            "Swing/thematic trading in liquid alts within universe. "
            "Build theme baskets and sector rotation (L2/AI/DeFi) within liquidity tiers. "
            "Maintain strict sizing (never outsized in illiquid assets). "
            "Plan unlock/supply events with research. "
            "Allocate limited capital to emerging alts (micro-caps or new sectors) for "
            "potentially high gains; strict exit if liquidity drops."
        ),
        backstory=(
            "Alt cycle expert with discipline in liquidity and position sizing. "
            "Understands how to profit from alt momentum without getting stuck "
            "in illiquidity. Quick to exit on narrative breaks."
        ),
        tools=tools,
        allow_delegation=False,
    )
