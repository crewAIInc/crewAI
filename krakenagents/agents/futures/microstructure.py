"""Futures Microstructure/Intraday Trading agents (39, 45-47) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_futures_market_tools, get_futures_execution_tools


def create_futures_microstructure_head_agent() -> Agent:
    """Maak Agent 39: Head of Microstructure/Intraday Futures.

    Eigenaar van intraday en orderflow strategieën op futures.
    Gebruikt heavy LLM voor microstructure analyse.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_market_tools() + get_futures_execution_tools()

    return create_heavy_agent(
        role="Head of Microstructure/Intraday Futures — Microstructure Trading",
        goal=(
            "Eigenaar intraday en orderflow strategieën op futures. "
            "Ontwikkel orderflow-gebaseerde entry/exit frameworks. "
            "Ontwerp liquidatie-niveau hunting strategieën. "
            "Coördineer met research over open interest dynamiek. "
            "Bouw playbooks voor high-volatility gebeurtenissen."
        ),
        backstory=(
            "Microstructure expert gespecialiseerd in crypto derivatives. "
            "Diepgaand begrip van perpetual orderflow en liquidatie cascades. "
            "Bekend om winstgevende intraday trading met strikte risicocontroles."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_intraday_i_agent() -> Agent:
    """Maak Agent 45: Intraday Trader Futures I (BTC/ETH Perps).

    Intraday trading op BTC/ETH perpetuals.
    Gebruikt light LLM voor uitvoering.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_market_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Intraday Trader Futures I (BTC/ETH Perps) — Microstructure Trading",
        goal=(
            "Intraday trading op BTC/ETH perpetuals. "
            "Handel liquidatie-niveau breakouts en absorptie. "
            "Gebruik funding rate als directioneel signaal. "
            "Strikte dagelijkse verlieslimieten met onmiddellijke cut. "
            "Schaal op tijdens high-vol macro gebeurtenissen."
        ),
        backstory=(
            "Intraday derivatives trader met BTC/ETH specialisatie. "
            "Expert in perpetual orderflow en funding dynamiek. "
            "Bekend om consistente intraday PnL met lage drawdowns."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_intraday_ii_agent() -> Agent:
    """Maak Agent 46: Intraday Trader Futures II (Alt Perps).

    Intraday trading op alt perpetuals.
    Gebruikt light LLM voor uitvoering.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_market_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Intraday Trader Futures II (Alt Perps) — Microstructure Trading",
        goal=(
            "Intraday trading op alt perpetuals. "
            "Focus op high-funding of high-OI alts. "
            "Handel momentum breakouts met funding bevestiging. "
            "Strikte liquiditeitsfilters voor alt perps. "
            "Snelle exits bij funding regime verandering."
        ),
        backstory=(
            "Intraday alt derivatives trader. "
            "Expert in het identificeren van high-alpha alt perp setups. "
            "Bekend om gedisciplineerde positiegrootte in minder liquide markten."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_futures_orderflow_agent() -> Agent:
    """Maak Agent 47: Orderflow Analyst/Trader Futures.

    Orderflow analyse en trading op futures.
    Gebruikt light LLM voor orderflow uitvoering.
    """
    tools = [
        TradeJournalTool(),
    ] + get_futures_market_tools() + get_futures_execution_tools()

    return create_light_agent(
        role="Orderflow Analyst/Trader Futures — Microstructure Trading",
        goal=(
            "Orderflow analyse en trading op futures. "
            "Monitor OI veranderingen, liquidatie niveaus en delta skew. "
            "Identificeer grote speler voetafdrukken in orderflow. "
            "Handel rond liquidatie cascade setups. "
            "Waarschuw desk voor significante orderflow veranderingen."
        ),
        backstory=(
            "Orderflow specialist met derivatives focus. "
            "Expert in het lezen van open interest en liquidatie heatmaps. "
            "Bekend om het identificeren van institutionele voetafdrukken in orderflow."
        ),
        tools=tools,
        allow_delegation=False,
    )
