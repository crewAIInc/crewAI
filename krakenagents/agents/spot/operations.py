"""Spot Operations agents (28-32) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import PnLTrackerTool, AlertSystemTool
from krakenagents.tools import get_spot_operations_tools


def create_spot_controller_agent() -> Agent:
    """Create Agent 28: Controller Spot.

    Financial control and reporting for spot desk.
    Uses light LLM for financial operations.
    """
    tools = [
        PnLTrackerTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Controller Spot — Operations",
        goal=(
            "Financial control and reporting for spot desk. "
            "Daily P&L reconciliation and NAV calculation. "
            "Fee tracking and cost allocation. "
            "Financial reporting and variance analysis. "
            "Audit support and documentation."
        ),
        backstory=(
            "Financial controller with trading desk experience. "
            "Expert in P&L reconciliation and fund accounting. "
            "Known for accurate and timely financial reporting."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_treasury_agent() -> Agent:
    """Create Agent 29: Treasury Spot.

    Cash and asset management for spot desk.
    Uses light LLM for treasury operations.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Treasury Spot — Operations",
        goal=(
            "Cash and asset management for spot desk. "
            "Monitor cash positions across venues. "
            "Manage deposits and withdrawals. "
            "Optimize asset deployment and idle cash. "
            "Coordinate with execution on funding needs."
        ),
        backstory=(
            "Treasury specialist with crypto exchange expertise. "
            "Expert in multi-venue cash management. "
            "Known for efficient asset deployment."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_security_agent() -> Agent:
    """Create Agent 30: Security Spot.

    Operational security for spot desk.
    Uses light LLM for security operations.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Security Spot — Operations",
        goal=(
            "Operational security for spot desk. "
            "Monitor for suspicious activity and anomalies. "
            "Enforce access controls and security procedures. "
            "Coordinate with Group Security on incidents. "
            "Maintain security documentation and training."
        ),
        backstory=(
            "Security specialist with trading operations experience. "
            "Expert in operational security and incident response. "
            "Known for maintaining secure trading environment."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_compliance_agent() -> Agent:
    """Create Agent 31: Compliance Spot.

    Regulatory compliance for spot desk.
    Uses light LLM for compliance operations.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Compliance Spot — Operations",
        goal=(
            "Regulatory compliance for spot desk. "
            "Monitor trading for compliance violations. "
            "Enforce restricted list and trading policies. "
            "Handle compliance queries and reporting. "
            "Coordinate with Group Compliance on issues."
        ),
        backstory=(
            "Compliance specialist with crypto trading expertise. "
            "Expert in trade surveillance and regulatory requirements. "
            "Known for effective compliance without trading friction."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_ops_agent() -> Agent:
    """Create Agent 32: Operations Support Spot.

    General operations support for spot desk.
    Uses light LLM for operations tasks.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Operations Support Spot — Operations",
        goal=(
            "General operations support for spot desk. "
            "Handle settlement and reconciliation breaks. "
            "Manage venue communications and issues. "
            "Support onboarding of new venues and assets. "
            "Maintain operational documentation and SOPs."
        ),
        backstory=(
            "Operations generalist with trading support experience. "
            "Expert in problem-solving and process improvement. "
            "Known for quick resolution of operational issues."
        ),
        tools=tools,
        allow_delegation=False,
    )
