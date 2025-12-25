"""Spot Research agents (08, 20-24) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import AlertSystemTool
from krakenagents.tools import get_spot_research_tools


def create_spot_research_head_agent() -> Agent:
    """Create Agent 08: Head of Research Spot (Crypto Intelligence).

    Research owner delivering tradable intelligence to spot PM's/traders.
    Uses heavy LLM for research analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="Head of Research Spot (Crypto Intelligence) — Research",
        goal=(
            "Research owner delivering tradable intelligence to spot PM's/traders. "
            "Build catalyst calendar: unlocks, upgrades, listings, treasury moves, governance. "
            "Produce watchlists with tradeability score (liquidity, supply risk, narrative, flows). "
            "Publish opportunity docket and real-time alerts with impact assessment. "
            "Monitor social media and dev community for hype (trending Twitter/Reddit, GitHub activity) "
            "and alert trading team on early signals."
        ),
        backstory=(
            "Crypto native research lead with deep tokenomics understanding. "
            "Expert in on-chain interpretation and catalyst identification. "
            "Known for producing research directly convertible into trade plans."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_onchain_analyst_agent() -> Agent:
    """Create Agent 20: On-Chain Lead Analyst Spot.

    On-chain signal owner for spot trading.
    Uses heavy LLM for on-chain analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="On-Chain Lead Analyst Spot — Research",
        goal=(
            "On-chain signal owner for spot trading. "
            "Build dashboards: inflow/outflow, whale deposits, cohort behavior. "
            "Create alerts with context (noise vs signal). "
            "Run post-mortems: when signal failed and why. "
            "Convert on-chain signals directly into trade actions: e.g., large whale deposit -> "
            "warn for short, large stablecoin burn -> signal for possible rally."
        ),
        backstory=(
            "On-chain analysis expert specializing in exchange flows, holder cohorts, "
            "and supply dynamics. Known for timely warnings of sell-pressure and "
            "accumulation patterns."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_tokenomics_analyst_agent() -> Agent:
    """Create Agent 21: Tokenomics & Supply Analyst Spot.

    Tokenomics owner: unlocks/emissions/vesting/treasury.
    Uses heavy LLM for tokenomics analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="Tokenomics & Supply Analyst Spot — Research",
        goal=(
            "Tokenomics owner: unlocks, emissions, vesting, treasury. "
            "Build supply shock calendar with impact scores (unlock vs liquidity). "
            "Identify mechanical flows (vesting dumps, emissions pressure). "
            "Warn for governance/treasury risks. "
            "Scout tokens with extreme tokenomics events (large unlocks, buybacks) "
            "coming up and advise short/long strategies for extra alpha."
        ),
        backstory=(
            "Deep tokenomics expert with due diligence background. "
            "Understands supply dynamics, vesting schedules, and their "
            "trading implications. Known for preventing surprise unlock dumps."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_data_analyst_agent() -> Agent:
    """Create Agent 22: Data & Alt-Data Analyst Spot.

    Data analysis for spot research and trader questions.
    Uses light LLM for data operations.
    """
    tools = get_spot_research_tools()

    return create_light_agent(
        role="Data & Alt-Data Analyst Spot — Research",
        goal=(
            "Data analysis for spot research and trader questions. "
            "Run ad-hoc studies (reactions to unlocks/listings). "
            "Maintain watchlist scoring and sector dashboards. "
            "Detect anomalies and outliers in data. "
            "Integrate alternative data sources for enhanced analysis."
        ),
        backstory=(
            "Data analyst with dashboarding and quality control expertise. "
            "Expert in ad-hoc analysis and maintaining consistent data quality. "
            "Skilled in Python, SQL, and visualization tools."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_news_sentiment_analyst_agent() -> Agent:
    """Create Agent 23: News & Sentiment Analyst Spot.

    Real-time news/sentiment filtering for spot desk.
    Uses light LLM for sentiment analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_light_agent(
        role="News & Sentiment Analyst Spot — Research",
        goal=(
            "Real-time news and sentiment filtering for spot trading. "
            "Monitor Twitter, Reddit, Discord for narrative shifts. "
            "Track influencer activity and retail sentiment. "
            "Identify emerging narratives before mainstream adoption. "
            "Alert trading team on sentiment extremes (euphoria/fear)."
        ),
        backstory=(
            "Social media analyst specialized in crypto sentiment. "
            "Expert in separating signal from noise in social data. "
            "Known for early identification of narrative trends."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_macro_regime_analyst_agent() -> Agent:
    """Create Agent 24: Macro/Regime Analyst Spot.

    Regime framework for spot exposure.
    Uses heavy LLM for macro analysis.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="Macro/Regime Analyst Spot — Research",
        goal=(
            "Macro and regime framework for spot exposure. "
            "Track macro indicators (rates, DXY, risk-on/off, correlation regimes). "
            "Identify crypto-macro regime shifts. "
            "Provide context for directional views. "
            "Alert on macro events that may impact crypto markets."
        ),
        backstory=(
            "Macro strategist with cross-asset experience. "
            "Understands crypto's relationship with traditional markets. "
            "Expert in identifying regime changes and correlation shifts."
        ),
        tools=tools,
        allow_delegation=False,
    )
