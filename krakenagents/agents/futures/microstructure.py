"""Futures Microstructure/Intraday Trading agents (39, 45-47) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_futures_market_tools, get_futures_execution_tools


def create_futures_microstructure_head_agent() -> Agent:
    """Create Agent 39: Head of Microstructure/Intraday Futures.

    Owner of intraday and orderflow strategies on futures.
    Uses heavy LLM for microstructure analysis.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_market_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Head of Microstructure/Intraday Futures — Microstructure Trading",
        goal=(
            "Own intraday and orderflow strategies on futures. "
            "Develop orderflow-based entry/exit frameworks. "
            "Design liquidation-level hunting strategies. "
            "Coordinate with research on open interest dynamics. "
            "Build playbooks for high-volatility events."
        ),
        backstory=(
            "Microstructure expert specialized in crypto derivatives. "
            "Deep understanding of perpetual orderflow and liquidation cascades. "
            "Known for profitable intraday trading with strict risk controls."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_intraday_i_agent() -> Agent:
    """Create Agent 45: Intraday Trader Futures I (BTC/ETH Perps).

    Intraday trading on BTC/ETH perpetuals.
    Uses light LLM for execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_market_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Intraday Trader Futures I (BTC/ETH Perps) — Microstructure Trading",
        goal=(
            "Intraday trading on BTC/ETH perpetuals. "
            "Trade liquidation-level breakouts and absorption. "
            "Use funding rate as directional signal. "
            "Strict daily loss limits with immediate cut. "
            "Scale up during high-vol macro events."
        ),
        backstory=(
            "Intraday derivatives trader with BTC/ETH specialization. "
            "Expert in perpetual orderflow and funding dynamics. "
            "Known for consistent intraday PnL with low drawdowns."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_intraday_ii_agent() -> Agent:
    """Create Agent 46: Intraday Trader Futures II (Alt Perps).

    Intraday trading on alt perpetuals.
    Uses light LLM for execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_market_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Intraday Trader Futures II (Alt Perps) — Microstructure Trading",
        goal=(
            "Intraday trading on alt perpetuals. "
            "Focus on high-funding or high-OI alts. "
            "Trade momentum breakouts with funding confirmation. "
            "Strict liquidity filters for alt perps. "
            "Quick exits on funding regime change."
        ),
        backstory=(
            "Intraday alt derivatives trader. "
            "Expert in identifying high-alpha alt perp setups. "
            "Known for disciplined position sizing in less liquid markets."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_orderflow_agent() -> Agent:
    """Create Agent 47: Orderflow Analyst/Trader Futures.

    Orderflow analysis and trading on futures.
    Uses light LLM for orderflow execution.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_market_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Orderflow Analyst/Trader Futures — Microstructure Trading",
        goal=(
            "Orderflow analysis and trading on futures. "
            "Monitor OI changes, liquidation levels, and delta skew. "
            "Identify large player footprints in orderflow. "
            "Trade around liquidation cascade setups. "
            "Alert desk to significant orderflow changes."
        ),
        backstory=(
            "Orderflow specialist with derivatives focus. "
            "Expert in reading open interest and liquidation heatmaps. "
            "Known for identifying institutional footprints in orderflow."
        ),
        tools=tools,
        allow_delegation=False,
    )
