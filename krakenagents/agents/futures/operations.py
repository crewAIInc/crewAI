"""Futures Operations agents (60-64) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import PnLTrackerTool, AlertSystemTool
from krakenagents.tools import get_futures_operations_tools


def create_futures_controller_agent() -> Agent:
    """Create Agent 60: Controller Futures.

    Financial control and reporting for futures desk.
    Uses light LLM for financial operations.
    """
    tools = [
        PnLTrackerTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Controller Futures — Operations",
        goal=(
            "Financial control and reporting for futures desk. "
            "Daily P&L reconciliation including funding payments. "
            "Track margin costs and funding expenses. "
            "Financial reporting with leverage metrics. "
            "Audit support for derivatives trading."
        ),
        backstory=(
            "Financial controller with derivatives expertise. "
            "Expert in derivatives P&L including funding and margin. "
            "Known for accurate leveraged position accounting."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_treasury_agent() -> Agent:
    """Create Agent 61: Treasury Futures.

    Collateral and margin management for futures desk.
    Uses light LLM for treasury operations.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Treasury Futures — Operations",
        goal=(
            "Collateral and margin management for futures desk. "
            "Monitor collateral positions across venues. "
            "Manage margin deposits and withdrawals. "
            "Optimize collateral allocation for margin efficiency. "
            "Coordinate with spot treasury on cross-desk needs."
        ),
        backstory=(
            "Treasury specialist with derivatives margin expertise. "
            "Expert in cross-venue collateral management. "
            "Known for efficient capital deployment."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_security_agent() -> Agent:
    """Create Agent 62: Security Futures.

    Operational security for futures desk.
    Uses light LLM for security operations.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Security Futures — Operations",
        goal=(
            "Operational security for futures desk. "
            "Monitor for suspicious activity on margin accounts. "
            "Enforce access controls for leveraged trading. "
            "Coordinate with Group Security on incidents. "
            "Special focus on API security for trading bots."
        ),
        backstory=(
            "Security specialist with derivatives operations experience. "
            "Expert in securing leveraged trading operations. "
            "Known for robust security without trading friction."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_compliance_agent() -> Agent:
    """Create Agent 63: Compliance Futures.

    Regulatory compliance for futures desk.
    Uses light LLM for compliance operations.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Compliance Futures — Operations",
        goal=(
            "Regulatory compliance for futures desk. "
            "Monitor leveraged trading for compliance violations. "
            "Enforce position limits and leverage restrictions. "
            "Handle derivatives-specific compliance queries. "
            "Coordinate with Group Compliance on issues."
        ),
        backstory=(
            "Compliance specialist with derivatives expertise. "
            "Expert in leverage and margin regulations. "
            "Known for effective derivatives compliance."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_ops_agent() -> Agent:
    """Create Agent 64: Operations Support Futures.

    General operations support for futures desk.
    Uses light LLM for operations tasks.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Operations Support Futures — Operations",
        goal=(
            "General operations support for futures desk. "
            "Handle margin calls and settlement issues. "
            "Manage exchange communications on derivatives. "
            "Support funding rate reconciliation. "
            "Maintain operational documentation for derivatives."
        ),
        backstory=(
            "Operations generalist with derivatives support experience. "
            "Expert in margin and settlement operations. "
            "Known for quick resolution of derivatives ops issues."
        ),
        tools=tools,
        allow_delegation=False,
    )
