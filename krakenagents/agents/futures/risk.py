"""Futures Risk Management agents (57-59) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import RiskDashboardTool, AlertSystemTool
from krakenagents.tools import get_futures_risk_tools


def create_futures_risk_monitor_agent() -> Agent:
    """Maak Agent 57: Real-Time Risk Monitor Futures.

    24/7 monitoring van futures risico metrics.
    Gebruikt light LLM voor monitoring taken.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_futures_risk_tools()

    return create_light_agent(
        role="Real-Time Risk Monitor Futures — Risk Management",
        goal=(
            "24/7 monitoring van futures risico metrics. "
            "Monitor leverage, margin gebruik en liquidatie afstand. "
            "Waarschuw onmiddellijk bij margin drempel schendingen. "
            "Volg funding blootstelling over posities. "
            "Escaleer naar CRO Futures bij kritieke waarschuwingen."
        ),
        backstory=(
            "Risico monitoring specialist met derivatives expertise. "
            "Expert in margin mechanismen en liquidatierisico. "
            "Bekend om snelle escalatie bij margin problemen."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_margin_analyst_agent() -> Agent:
    """Maak Agent 58: Margin & Collateral Analyst Futures.

    Margin en collateral optimalisatie voor futures.
    Gebruikt light LLM voor margin analyse.
    """
    tools = [
        RiskDashboardTool(),
    ] + get_futures_risk_tools()

    return create_light_agent(
        role="Margin & Collateral Analyst Futures — Risk Management",
        goal=(
            "Optimaliseer margin en collateral voor futures posities. "
            "Monitor margin gebruik over venues. "
            "Optimaliseer collateral inzet voor margin efficiëntie. "
            "Volg cross-margin voordelen en risico's. "
            "Coördineer met treasury over collateral bewegingen."
        ),
        backstory=(
            "Margin specialist met derivatives exchange expertise. "
            "Expert in cross-margin optimalisatie. "
            "Bekend om het maximaliseren van kapitaal efficiëntie."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_liquidation_agent() -> Agent:
    """Maak Agent 59: Liquidation Risk Specialist Futures.

    Specialist in het monitoren en voorkomen van liquidaties.
    Gebruikt light LLM voor liquidatie analyse.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_futures_risk_tools()

    return create_light_agent(
        role="Liquidation Risk Specialist Futures — Risk Management",
        goal=(
            "Monitor en voorkom liquidaties over futures posities. "
            "Volg liquidatie prijzen en afstand voor alle posities. "
            "Modelleer liquidatie cascade scenario's. "
            "Coördineer nood positie reducties. "
            "Waarschuw bij naderende liquidatie drempels."
        ),
        backstory=(
            "Liquidatie risico specialist met crisis ervaring. "
            "Expert in liquidatie mechanismen en cascade dynamiek. "
            "Bekend om het voorkomen van onnodige liquidaties."
        ),
        tools=tools,
        allow_delegation=False,
    )
