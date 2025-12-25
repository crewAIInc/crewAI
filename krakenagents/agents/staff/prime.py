"""STAFF-07: Head of Prime, Venues & Liquidity Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools import (
    get_spot_market_tools,
    get_futures_market_tools,
)


def create_prime_agent() -> Agent:
    """Create STAFF-07 Head of Prime, Venues & Liquidity Agent.

    Responsible for:
    - Venue selection and health monitoring
    - Liquidity access and fee tier optimization
    - Venue concentration risk management
    - Prime broker and OTC relationships

    Reports to: STAFF-00 (CEO)
    Uses light LLM for venue/liquidity operations.
    """
    # Market tools for monitoring venue health and liquidity
    tools = get_spot_market_tools() + get_futures_market_tools()

    return create_light_agent(
        role="Head of Prime, Venues & Liquidity — Venue Selection and Liquidity Access",
        goal=(
            "Manage venue selection, liquidity access, fee tiers, and concentration risk. "
            "Maintain venue scorecards and propose limits (with CRO). Optimize liquidity and fees "
            "per venue. Mitigate concentration risk through diversification planning. "
            "Invest in low-latency connectivity (co-location, dedicated lines) to key venues. "
            "Maintain multi-venue relationships (prime brokers, OTC desks) for deep liquidity "
            "and fast execution of large orders."
        ),
        backstory=(
            "Market structure expert with deep understanding of crypto exchange ecosystems, "
            "prime brokerage, and OTC markets. Strong relationships with major venues and "
            "liquidity providers. Expert in fee optimization and execution quality analysis. "
            "Known for securing favorable trading terms and managing venue risk through "
            "diversification. Technical background in low-latency trading infrastructure."
        ),
        tools=tools,
        allow_delegation=False,
    )
