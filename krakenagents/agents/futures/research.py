"""Futures Research agents (40, 52-56) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import AlertSystemTool
from krakenagents.tools import get_futures_research_tools


def create_futures_research_head_agent() -> Agent:
    """Create Agent 40: Head of Research Futures.

    Research owner for derivatives/futures intelligence.
    Uses heavy LLM for research analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Head of Research Futures — Research",
        goal=(
            "Research owner for derivatives/futures intelligence. "
            "Build funding rate forecast models. "
            "Monitor cross-exchange basis and funding differentials. "
            "Track open interest dynamics and positioning. "
            "Produce tradable derivatives research for the desk."
        ),
        backstory=(
            "Derivatives research lead with quantitative background. "
            "Expert in funding rate modeling and OI analysis. "
            "Known for producing actionable derivatives research."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_funding_analyst_agent() -> Agent:
    """Create Agent 52: Funding Rate Analyst Futures.

    Funding rate analysis and prediction.
    Uses heavy LLM for funding analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Funding Rate Analyst Futures — Research",
        goal=(
            "Analyze and predict funding rates. "
            "Build funding rate forecast models. "
            "Monitor funding across venues and instruments. "
            "Identify funding regime changes. "
            "Alert on extreme funding conditions."
        ),
        backstory=(
            "Funding rate specialist with modeling expertise. "
            "Expert in perpetual swap mechanics and funding dynamics. "
            "Known for accurate funding regime predictions."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_basis_analyst_agent() -> Agent:
    """Create Agent 53: Basis & Term Structure Analyst Futures.

    Basis and term structure analysis.
    Uses heavy LLM for basis analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Basis & Term Structure Analyst Futures — Research",
        goal=(
            "Analyze basis and term structure dynamics. "
            "Monitor spot-futures basis across instruments. "
            "Track calendar spread opportunities. "
            "Identify convergence and divergence patterns. "
            "Alert on significant basis dislocations."
        ),
        backstory=(
            "Term structure specialist with derivatives background. "
            "Expert in basis analysis and convergence trading. "
            "Known for identifying profitable basis opportunities."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_quant_analyst_agent() -> Agent:
    """Create Agent 54: Quant Analyst Futures.

    Quantitative analysis for derivatives strategies.
    Uses heavy LLM for quant analysis.
    """
    tools = get_futures_research_tools()

    return create_heavy_agent(
        role="Quant Analyst Futures — Research",
        goal=(
            "Quantitative analysis for derivatives strategies. "
            "Build statistical models for derivatives signals. "
            "Backtest and validate strategy ideas. "
            "Develop risk models for leveraged positions. "
            "Support systematic strategy development."
        ),
        backstory=(
            "Quantitative analyst specialized in derivatives. "
            "Expert in statistical modeling and backtesting. "
            "Known for rigorous validation of strategy ideas."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_macro_analyst_agent() -> Agent:
    """Create Agent 55: Macro Analyst Futures.

    Macro analysis for derivatives trading context.
    Uses heavy LLM for macro analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_heavy_agent(
        role="Macro Analyst Futures — Research",
        goal=(
            "Macro analysis for derivatives trading context. "
            "Track macro indicators affecting crypto derivatives. "
            "Monitor correlation between crypto and traditional markets. "
            "Identify macro regime shifts. "
            "Alert on macro events impacting derivatives."
        ),
        backstory=(
            "Macro strategist with derivatives focus. "
            "Expert in crypto-macro relationships. "
            "Known for timely macro regime change identification."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_flow_analyst_agent() -> Agent:
    """Create Agent 56: Flow & Positioning Analyst Futures.

    Flow and positioning analysis for derivatives.
    Uses light LLM for flow analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_research_tools()

    return create_light_agent(
        role="Flow & Positioning Analyst Futures — Research",
        goal=(
            "Analyze flow and positioning in derivatives markets. "
            "Track open interest changes and liquidations. "
            "Monitor long/short ratios and crowding. "
            "Identify significant positioning changes. "
            "Alert on extreme positioning or liquidation risk."
        ),
        backstory=(
            "Flow analyst specialized in crypto derivatives. "
            "Expert in OI analysis and positioning dynamics. "
            "Known for identifying crowded trades and liquidation risks."
        ),
        tools=tools,
        allow_delegation=False,
    )
