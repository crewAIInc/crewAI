"""Spot Executie agenten (09, 16, 17) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_light_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_spot_execution_tools, get_spot_market_tools


def create_spot_execution_head_agent() -> Agent:
    """Maak Agent 09: Hoofd Executie Spot.

    Vermindert slippage/fees en standaardiseert executie voor spot.
    Gebruikt licht LLM voor executie optimalisatie.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_execution_tools() + get_spot_market_tools()

    return create_light_agent(
        role="Hoofd Executie Spot — Executie",
        goal=(
            "Verminder slippage/fees en standaardiseer executie voor spot. "
            "Stel executie KPI's in (implementation shortfall, reject rate, adverse selection). "
            "Ontwikkel maker/taker beleid, routing regels en grote-order playbooks. "
            "Verbeter continu fills en kosten over venues. "
            "Integreer geautomatiseerde executie algo's (TWAP/VWAP) en verken dark liquidity bronnen "
            "om grote orders uit te voeren zonder markt impact."
        ),
        backstory=(
            "Executie specialist met CEX/spot order type expertise. "
            "Diep begrip van fee niveaus en microstructuur. "
            "Bekend om het bouwen van stabiele executie standaarden die alpha lekkage verminderen."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_intraday_btc_agent() -> Agent:
    """Maak Agent 16: Intraday Orderflow Trader Spot I (BTC/ETH).

    Intraday spot BTC/ETH trading met orderflow bevestiging.
    Gebruikt licht LLM voor intraday executie.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_execution_tools() + get_spot_market_tools()

    return create_light_agent(
        role="Intraday Orderflow Trader Spot I (BTC/ETH) — Executie",
        goal=(
            "Intraday spot BTC/ETH trading met orderflow bevestiging. "
            "Speel setups: breakout validatie, absorptie, liquiditeit muren (alleen liquide paren). "
            "Journal met setup tags en executie notities. "
            "Handhaaf stop discipline (geen stops verplaatsen buiten playbook). "
            "Verhoog positie grootte licht wanneer 'in the zone' en markt trend bevestigt, "
            "voor extra PnL (maar respecteer dagelijks verlies limiet)."
        ),
        backstory=(
            "Tape/DOM expert met strikte dagelijkse verlies limieten. "
            "Gedisciplineerde intraday trader gefocust op orderflow bevestiging. "
            "Bekend om consistente intraday PnL met lage drawdowns."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_intraday_majors_agent() -> Agent:
    """Maak Agent 17: Intraday Orderflow Trader Spot II (Majors).

    Intraday majors trading met momentum/range edges.
    Gebruikt licht LLM voor intraday executie.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_execution_tools() + get_spot_market_tools()

    return create_light_agent(
        role="Intraday Orderflow Trader Spot II (Majors) — Executie",
        goal=(
            "Intraday majors trading met momentum en range edges. "
            "Respecteer no-trade vensters tijdens dunne liquiditeit. "
            "Coördineer grotere entries/exits met executie desk. "
            "Dagelijkse zelf-review en desk review met Hoofd Trading. "
            "Ga vol in tijdens macro events intraday (CPI, FOMC) wanneer liquiditeit hoog is "
            "voor duidelijke bewegingen; vermijd overtrading na de spike."
        ),
        backstory=(
            "Intraday trader die begrijpt wanneer orderbook misleidend is. "
            "Expert in het herkennen van dunne-liquiditeit vallen. "
            "Bekend om lage overtrading en hoge signaal-ruis verhouding."
        ),
        tools=tools,
        allow_delegation=False,
    )
