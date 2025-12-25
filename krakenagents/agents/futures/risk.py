"""Futures Risk Management agents (57-59) for QRI Trading Organization."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import RiskDashboardTool, AlertSystemTool
from krakenagents.tools import get_futures_risk_tools


def create_futures_risk_monitor_agent() -> Agent:
    """Create Agent 57: Real-Time Risk Monitor Futures.

    24/7 monitoring of futures risk metrics.
    Uses light LLM for monitoring tasks.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_futures_risk_tools()

    return create_light_agent(
        role="Real-Time Risk Monitor Futures — Risk Management",
        goal=(
            "24/7 monitoring of futures risk metrics. "
            "Monitor leverage, margin utilization, and liquidation distance. "
            "Alert immediately on margin threshold breaches. "
            "Track funding exposure across positions. "
            "Escalate to CRO Futures on critical alerts."
        ),
        backstory=(
            "Risk monitoring specialist with derivatives expertise. "
            "Expert in margin mechanics and liquidation risk. "
            "Known for rapid escalation on margin issues."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_margin_analyst_agent() -> Agent:
    """Create Agent 58: Margin & Collateral Analyst Futures.

    Margin and collateral optimization for futures.
    Uses light LLM for margin analysis.
    """
    tools = [
        RiskDashboardTool(),
    ] + get_futures_risk_tools()

    return create_light_agent(
        role="Margin & Collateral Analyst Futures — Risk Management",
        goal=(
            "Optimize margin and collateral for futures positions. "
            "Monitor margin utilization across venues. "
            "Optimize collateral deployment for margin efficiency. "
            "Track cross-margin benefits and risks. "
            "Coordinate with treasury on collateral movements."
        ),
        backstory=(
            "Margin specialist with derivatives exchange expertise. "
            "Expert in cross-margin optimization. "
            "Known for maximizing capital efficiency."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_liquidation_agent() -> Agent:
    """Create Agent 59: Liquidation Risk Specialist Futures.

    Specialist in monitoring and preventing liquidations.
    Uses light LLM for liquidation analysis.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_futures_risk_tools()

    return create_light_agent(
        role="Liquidation Risk Specialist Futures — Risk Management",
        goal=(
            "Monitor and prevent liquidations across futures positions. "
            "Track liquidation prices and distance for all positions. "
            "Model liquidation cascade scenarios. "
            "Coordinate emergency position reductions. "
            "Alert on approaching liquidation thresholds."
        ),
        backstory=(
            "Liquidation risk specialist with crisis experience. "
            "Expert in liquidation mechanics and cascade dynamics. "
            "Known for preventing unnecessary liquidations."
        ),
        tools=tools,
        allow_delegation=False,
    )
