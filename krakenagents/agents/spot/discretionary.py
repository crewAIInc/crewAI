"""Spot Discretionaire Trading agenten (06, 13, 14) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent
from krakenagents.tools.internal import TradeJournalTool
from krakenagents.tools import get_spot_research_tools, get_spot_execution_tools


def create_spot_discretionary_head_agent() -> Agent:
    """Maak Agent 06: Hoofd Discretionair Spot (Thema's & Swing).

    Eigenaar van discretionair spot swing/thematisch boek.
    Gebruikt zwaar LLM voor thesis-gedreven beslissingen.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_research_tools() + get_spot_execution_tools()

    return create_heavy_agent(
        role="Hoofd Discretionair Spot (Thema's & Swing) — Discretionaire Trading",
        goal=(
            "Eigenaar van discretionair spot swing/thematisch boek. "
            "Bouw thesis-gedreven trades (dagen-weken) met strikte invalidatie. "
            "Integreer onderzoek: tokenomics/unlocks, flows, catalysatoren. "
            "Beheer posities: schaal in/uit, trailing, winst bescherming. "
            "Scout niche tokens/narratieven vroeg en neem kleine posities (bewust hoog risico) "
            "voor potentieel grote winsten als thesis uitkomt."
        ),
        backstory=(
            "Ervaren swing trader met diepe marktstructuur kennis. "
            "Expert in risico/reward analyse en thema rotatie. Bekend om "
            "het bouwen van grotere R-trades met gecontroleerde drawdowns en duidelijke exits."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_swing_majors_agent() -> Agent:
    """Maak Agent 13: Discretionaire Swing Trader Spot I (Majors).

    Swing trader voor BTC/ETH en top-liquide coins.
    Gebruikt zwaar LLM voor trade planning.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_research_tools() + get_spot_execution_tools()

    return create_heavy_agent(
        role="Discretionaire Swing Trader Spot I (Majors) — Discretionaire Trading",
        goal=(
            "Swing trade BTC/ETH en top-liquide coins. "
            "Plan trend voortzetting/pullback trades met invalidatie niveaus. "
            "Combineer niveaus met flows/volume (geen indicator blindheid). "
            "Bouw scenario trade plannen (basis/bull/bear). "
            "Laat winnaars lopen: verhoog positie of verbreed trailing stop wanneer trade "
            "overtuigend wint om trend capture te maximaliseren (behoud stop discipline)."
        ),
        backstory=(
            "Multi-timeframe trader expert in marktstructuur en risico/reward. "
            "Gebruikt flow en volume analyse naast technische niveaus. "
            "Bouwt herhaalbare playbooks met consistente executie."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_swing_alts_agent() -> Agent:
    """Maak Agent 14: Discretionaire Swing Trader Spot II (Alts/Thema's).

    Swing/thematische trader in liquide alts binnen universum.
    Gebruikt zwaar LLM voor thema analyse.
    """
    tools = [
        TradeJournalTool(),
    ] + get_spot_research_tools() + get_spot_execution_tools()

    return create_heavy_agent(
        role="Discretionaire Swing Trader Spot II (Alts/Thema's) — Discretionaire Trading",
        goal=(
            "Swing/thematische trading in liquide alts binnen universum. "
            "Bouw thema baskets en sector rotatie (L2/AI/DeFi) binnen liquiditeit niveaus. "
            "Handhaaf strikte sizing (nooit te groot in illiquide assets). "
            "Plan unlock/supply events met onderzoek. "
            "Alloceer beperkt kapitaal naar opkomende alts (micro-caps of nieuwe sectoren) voor "
            "potentieel hoge winsten; strikte exit als liquiditeit daalt."
        ),
        backstory=(
            "Alt cycle expert met discipline in liquiditeit en positie sizing. "
            "Begrijpt hoe te profiteren van alt momentum zonder vast te lopen "
            "in illiquiditeit. Snel met exiten bij narratief breuken."
        ),
        tools=tools,
        allow_delegation=False,
    )
