"""Spot Risk Management agents (19, 25-27) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import RiskDashboardTool, AlertSystemTool
from krakenagents.tools import get_spot_risk_tools


def create_spot_inventory_coordinator_agent() -> Agent:
    """Create Agent 19: Inventory & Risk Coordinator Spot.

    Coordinator between traders and risk for spot inventory/exposure.
    Uses light LLM for coordination tasks.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_risk_tools()

    return create_light_agent(
        role="Inventory & Risk Coordinator Spot — Risk Management",
        goal=(
            "Coordinate between traders and risk for spot inventory/exposure. "
            "Daily inventory checks: concentrations, liquidity tiers, exit readiness. "
            "Start inter-desk hedge request if hedge only possible via futures "
            "(without trading futures directly). "
            "Signal mismatch between exposure and regime. "
            "Don't hedge too early: let limited overexposure run if market is favorable; "
            "hedge only when risk asymmetry increases, for better risk/reward."
        ),
        backstory=(
            "Risk-aware trader/analyst focused on concentration and liquidity. "
            "Expert in coordinating between trading and risk functions. "
            "Known for reducing unintended beta/concentration risk."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_risk_monitor_agent() -> Agent:
    """Create Agent 25: Real-Time Risk Monitor Spot.

    24/7 monitoring of spot risk metrics and thresholds.
    Uses light LLM for monitoring tasks.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_risk_tools()

    return create_light_agent(
        role="Real-Time Risk Monitor Spot — Risk Management",
        goal=(
            "24/7 monitoring of spot risk metrics and thresholds. "
            "Monitor position limits, exposure caps, and drawdown levels. "
            "Alert immediately on threshold breaches. "
            "Escalate to CRO Spot on critical alerts. "
            "Track venue health and liquidity conditions."
        ),
        backstory=(
            "Risk monitoring specialist with attention to detail. "
            "Expert in real-time risk surveillance and alert management. "
            "Known for rapid escalation on critical issues."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_limits_officer_agent() -> Agent:
    """Create Agent 26: Limits & Controls Officer Spot.

    Maintains and enforces trading limits and controls.
    Uses light LLM for limits management.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_spot_risk_tools()

    return create_light_agent(
        role="Limits & Controls Officer Spot — Risk Management",
        goal=(
            "Maintain and enforce trading limits and controls for spot. "
            "Document and update limit frameworks. "
            "Process limit change requests with proper approvals. "
            "Ensure limit enforcement in trading systems. "
            "Regular limit reviews and calibration."
        ),
        backstory=(
            "Controls specialist with trading limits expertise. "
            "Expert in limit frameworks and governance. "
            "Known for maintaining consistent limit enforcement."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_margin_analyst_agent() -> Agent:
    """Create Agent 27: Margin & Collateral Analyst Spot.

    Monitors margin and collateral for spot positions.
    Uses light LLM for margin analysis.
    """
    tools = [
        RiskDashboardTool(),
    ] + get_spot_risk_tools()

    return create_light_agent(
        role="Margin & Collateral Analyst Spot — Risk Management",
        goal=(
            "Monitor margin and collateral for spot positions. "
            "Track margin utilization across venues. "
            "Optimize collateral allocation. "
            "Alert on margin calls or low buffer situations. "
            "Coordinate with treasury on collateral movements."
        ),
        backstory=(
            "Margin specialist with exchange margin model expertise. "
            "Expert in collateral optimization and margin efficiency. "
            "Known for preventing margin-related trading disruptions."
        ),
        tools=tools,
        allow_delegation=False,
    )
