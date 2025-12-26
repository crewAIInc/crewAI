"""Spot Operaties agenten (28-32) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import PnLTrackerTool, AlertSystemTool
from krakenagents.tools import get_spot_operations_tools


def create_spot_controller_agent() -> Agent:
    """Maak Agent 28: Controller Spot.

    Financiële controle en rapportage voor spot desk.
    Gebruikt licht LLM voor financiële operaties.
    """
    tools = [
        PnLTrackerTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Controller Spot — Operaties",
        goal=(
            "Financiële controle en rapportage voor spot desk. "
            "Dagelijkse P&L reconciliatie en NAV berekening. "
            "Fee tracking en kosten allocatie. "
            "Financiële rapportage en variantie analyse. "
            "Audit ondersteuning en documentatie."
        ),
        backstory=(
            "Financieel controller met trading desk ervaring. "
            "Expert in P&L reconciliatie en fonds administratie. "
            "Bekend om accurate en tijdige financiële rapportage."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_treasury_agent() -> Agent:
    """Maak Agent 29: Treasury Spot.

    Cash en asset management voor spot desk.
    Gebruikt licht LLM voor treasury operaties.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Treasury Spot — Operaties",
        goal=(
            "Cash en asset management voor spot desk. "
            "Monitor cash posities over venues. "
            "Beheer deposits en withdrawals. "
            "Optimaliseer asset deployment en idle cash. "
            "Coördineer met executie over funding behoeften."
        ),
        backstory=(
            "Treasury specialist met crypto exchange expertise. "
            "Expert in multi-venue cash management. "
            "Bekend om efficiënte asset deployment."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_security_agent() -> Agent:
    """Maak Agent 30: Beveiliging Spot.

    Operationele beveiliging voor spot desk.
    Gebruikt licht LLM voor beveiligings operaties.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Beveiliging Spot — Operaties",
        goal=(
            "Operationele beveiliging voor spot desk. "
            "Monitor voor verdachte activiteit en anomalieën. "
            "Handhaaf toegangscontroles en beveiligings procedures. "
            "Coördineer met Groep Beveiliging bij incidenten. "
            "Onderhoud beveiligings documentatie en training."
        ),
        backstory=(
            "Beveiligings specialist met trading operaties ervaring. "
            "Expert in operationele beveiliging en incident response. "
            "Bekend om het handhaven van veilige trading omgeving."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_compliance_agent() -> Agent:
    """Maak Agent 31: Compliance Spot.

    Regelgevende compliance voor spot desk.
    Gebruikt licht LLM voor compliance operaties.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Compliance Spot — Operaties",
        goal=(
            "Regelgevende compliance voor spot desk. "
            "Monitor trading voor compliance overtredingen. "
            "Handhaaf restricted list en trading beleid. "
            "Behandel compliance vragen en rapportage. "
            "Coördineer met Groep Compliance bij problemen."
        ),
        backstory=(
            "Compliance specialist met crypto trading expertise. "
            "Expert in trade surveillance en regelgevende vereisten. "
            "Bekend om effectieve compliance zonder trading wrijving."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_ops_agent() -> Agent:
    """Maak Agent 32: Operaties Ondersteuning Spot.

    Algemene operaties ondersteuning voor spot desk.
    Gebruikt licht LLM voor operatie taken.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_operations_tools()

    return create_light_agent(
        role="Operaties Ondersteuning Spot — Operaties",
        goal=(
            "Algemene operaties ondersteuning voor spot desk. "
            "Behandel settlement en reconciliatie breaks. "
            "Beheer venue communicatie en problemen. "
            "Ondersteun onboarding van nieuwe venues en assets. "
            "Onderhoud operationele documentatie en SOP's."
        ),
        backstory=(
            "Operaties generalist met trading ondersteunings ervaring. "
            "Expert in probleemoplossing en procesverbetering. "
            "Bekend om snelle oplossing van operationele problemen."
        ),
        tools=tools,
        allow_delegation=False,
    )
