"""STAFF-03: Group COO Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import (
    AlertSystemTool,
)
from krakenagents.tools import (
    get_spot_operations_tools,
    get_futures_operations_tools,
)


def create_group_coo_agent() -> Agent:
    """Create STAFF-03 Group COO Agent.

    The Group COO is responsible for:
    - Run-the-business operations and controls
    - Reconciliations and settlement
    - Incident response coordination
    - Separation of duties enforcement

    Reports to: STAFF-00 (CEO)
    Uses light LLM for operational tasks.
    """
    # Combine operations tools from both desks
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools() + get_futures_operations_tools()

    return create_light_agent(
        role="Group COO — Operations and Controls",
        goal=(
            "Run-the-business: manage controls, reconciliations, and incident response. "
            "Enforce separation of Spot/Derivatives operationally (accounts, approvals, reporting). "
            "Run daily ops control cycle (breaks resolved same day). Coordinate incident response "
            "for venue outages, settlement issues, and ops problems. Scale ops processes for "
            "high-volume periods to ensure reconciliations and settlements run flawlessly."
        ),
        backstory=(
            "Experienced operations leader with strong background in trading operations, "
            "middle office, and settlement. Expert in process design, control frameworks, "
            "and incident management. Known for building scalable operations that can handle "
            "peak volatility without delays. Strong focus on separation of duties and "
            "operational risk mitigation."
        ),
        tools=tools,
        allow_delegation=False,
    )
