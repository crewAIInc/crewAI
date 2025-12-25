"""Agents module for QRI Trading Organization.

Contains 74 agents total:
- 10 STAFF agents (Group Executive Board)
- 32 Spot desk agents
- 32 Futures desk agents
"""

from krakenagents.agents.base import create_agent, create_heavy_agent, create_light_agent
from krakenagents.agents.staff import get_all_staff_agents
from krakenagents.agents.spot import get_all_spot_agents
from krakenagents.agents.futures import get_all_futures_agents

__all__ = [
    "create_agent",
    "create_heavy_agent",
    "create_light_agent",
    "get_all_staff_agents",
    "get_all_spot_agents",
    "get_all_futures_agents",
    "get_all_agents",
]


def get_all_agents() -> list:
    """Create and return all 74 agents in the organization."""
    return (
        get_all_staff_agents()
        + get_all_spot_agents()
        + get_all_futures_agents()
    )
