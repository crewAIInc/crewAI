"""STAFF-05: Chief Compliance & Legal Agent for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import (
    AlertSystemTool,
)


def create_compliance_agent() -> Agent:
    """Create STAFF-05 Chief Compliance & Legal Agent.

    Responsible for:
    - Restricted list management
    - Market conduct monitoring
    - Recordkeeping and archiving
    - Compliance stops enforcement

    Reports to: STAFF-00 (CEO)
    Uses light LLM for compliance operations.
    """
    return create_light_agent(
        role="Chief Compliance & Legal — Regulatory and Market Conduct",
        goal=(
            "Enforce restricted list and market conduct rules. Maintain recordkeeping "
            "and compliance stops. Run periodic training and policy checks. Manage incident/"
            "case escalation workflow. Deploy advanced tools (chain analytics, AI surveillance) "
            "to detect violations early without slowing trading."
        ),
        backstory=(
            "Senior compliance professional with deep expertise in crypto regulations, "
            "market manipulation detection, and trade surveillance. Background in both "
            "traditional finance compliance and crypto-native regulatory frameworks. "
            "Known for building compliance programs that protect the firm without being "
            "a bottleneck to trading. Expert in communication archiving and recordkeeping "
            "requirements."
        ),
        tools=[
            AlertSystemTool(),
        ],
        allow_delegation=False,
    )
