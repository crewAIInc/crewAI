"""Futures Operations agents (60-64) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import PnLTrackerTool, AlertSystemTool
from krakenagents.tools import get_futures_operations_tools


def create_futures_controller_agent() -> Agent:
    """Maak Agent 60: Controller Futures.

    Financiële controle en rapportage voor futures desk.
    Gebruikt light LLM voor financiële operaties.
    """
    tools = [
        PnLTrackerTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Controller Futures — Operations",
        goal=(
            "Financiële controle en rapportage voor futures desk. "
            "Dagelijkse P&L reconciliatie inclusief funding betalingen. "
            "Volg margin kosten en funding uitgaven. "
            "Financiële rapportage met leverage metrics. "
            "Audit ondersteuning voor derivatives trading."
        ),
        backstory=(
            "Financiële controller met derivatives expertise. "
            "Expert in derivatives P&L inclusief funding en margin. "
            "Bekend om nauwkeurige leveraged positie boekhouding."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_treasury_agent() -> Agent:
    """Maak Agent 61: Treasury Futures.

    Collateral en margin management voor futures desk.
    Gebruikt light LLM voor treasury operaties.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Treasury Futures — Operations",
        goal=(
            "Collateral en margin management voor futures desk. "
            "Monitor collateral posities over venues. "
            "Beheer margin deposits en withdrawals. "
            "Optimaliseer collateral allocatie voor margin efficiëntie. "
            "Coördineer met spot treasury voor cross-desk behoeften."
        ),
        backstory=(
            "Treasury specialist met derivatives margin expertise. "
            "Expert in cross-venue collateral management. "
            "Bekend om efficiënte kapitaalinzet."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_security_agent() -> Agent:
    """Maak Agent 62: Security Futures.

    Operationele beveiliging voor futures desk.
    Gebruikt light LLM voor security operaties.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Security Futures — Operations",
        goal=(
            "Operationele beveiliging voor futures desk. "
            "Monitor voor verdachte activiteit op margin accounts. "
            "Handhaaf toegangscontroles voor leveraged trading. "
            "Coördineer met Groep Security bij incidenten. "
            "Speciale focus op API beveiliging voor trading bots."
        ),
        backstory=(
            "Security specialist met derivatives operaties ervaring. "
            "Expert in het beveiligen van leveraged trading operaties. "
            "Bekend om robuuste beveiliging zonder trading wrijving."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_compliance_agent() -> Agent:
    """Maak Agent 63: Compliance Futures.

    Regelgevende compliance voor futures desk.
    Gebruikt light LLM voor compliance operaties.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Compliance Futures — Operations",
        goal=(
            "Regelgevende compliance voor futures desk. "
            "Monitor leveraged trading voor compliance overtredingen. "
            "Handhaaf positielimieten en leverage restricties. "
            "Behandel derivatives-specifieke compliance vragen. "
            "Coördineer met Groep Compliance bij problemen."
        ),
        backstory=(
            "Compliance specialist met derivatives expertise. "
            "Expert in leverage en margin regelgeving. "
            "Bekend om effectieve derivatives compliance."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_ops_agent() -> Agent:
    """Maak Agent 64: Operations Support Futures.

    Algemene operaties ondersteuning voor futures desk.
    Gebruikt light LLM voor operationele taken.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_operations_tools()

    return create_light_agent(
        role="Operations Support Futures — Operations",
        goal=(
            "Algemene operaties ondersteuning voor futures desk. "
            "Behandel margin calls en settlement problemen. "
            "Beheer exchange communicatie over derivatives. "
            "Ondersteun funding rate reconciliatie. "
            "Onderhoud operationele documentatie voor derivatives."
        ),
        backstory=(
            "Operations generalist met derivatives ondersteuning ervaring. "
            "Expert in margin en settlement operaties. "
            "Bekend om snelle oplossing van derivatives ops problemen."
        ),
        tools=tools,
        allow_delegation=False,
    )
