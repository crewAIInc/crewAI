"""Futures Desk Leadership agents (33-36) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import (
    RiskDashboardTool,
    PnLTrackerTool,
    AlertSystemTool,
    TradeJournalTool,
)
from krakenagents.tools import get_futures_leadership_tools, get_futures_risk_tools


def create_futures_cio_agent() -> Agent:
    """Maak Agent 33: CIO Futures / Portfolio Manager.

    Verantwoordelijk voor Futures PnL, allocatie en strategiekeuzes.
    Rapporteert aan: CEO (hiërarchisch), Groep CIO (functioneel)
    Gebruikt heavy LLM voor complexe portefeuille beslissingen.
    """
    tools = [
        RiskDashboardTool(),
        PnLTrackerTool(),
    ] + get_futures_leadership_tools()

    return create_heavy_agent(
        role="CIO Futures / Portfolio Manager — Futures Desk Leadership",
        goal=(
            "Ultieme verantwoordelijkheid voor Futures PnL, allocatie en strategiekeuzes. "
            "Definieer verhandelbare instrumenten, blootstellingslimieten en allocatie per strategie. "
            "Stel risicobudgetten in per pod (systematisch/carry/microstructure/swing). "
            "Voer maandelijkse allocatie en stop/schaal beslissingen uit op basis van data. "
            "Optimaliseer leverage gebruik binnen risicolimieten voor verbeterde rendementen."
        ),
        backstory=(
            "10+ jaar derivatives/futures trading ervaring met bewezen trackrecord. "
            "Sterk in portefeuille constructie met leverage overwegingen. "
            "Expert in funding rate dynamiek, basis trading en perpetual mechanics. "
            "Data-gedreven beslisser die winnende strategieën opschaalt."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_head_trading_agent() -> Agent:
    """Maak Agent 34: Head of Trading Futures.

    Dagelijkse leiding van de futures trading floor.
    Rapporteert aan: CEO (hiërarchisch), Groep CIO/COO (functioneel)
    Gebruikt heavy LLM voor complexe trading beslissingen.
    """
    tools = [
        TradeJournalTool(),
        PnLTrackerTool(),
        AlertSystemTool(),
    ] + get_futures_leadership_tools()

    return create_heavy_agent(
        role="Head of Trading Futures — Futures Desk Leadership",
        goal=(
            "Dagelijkse leiding van de futures trading floor. "
            "Voer dagelijkse desk briefing uit met focuslijst, funding verwachtingen en risicomodus. "
            "Monitor playbook discipline en positiegrootte. "
            "Voer post-trade reviews uit met nadruk op leverage gebruik. "
            "Coördineer met spot desk voor cross-desk kansen."
        ),
        backstory=(
            "Ex-derivatives desk lead met sterke procesoriëntatie. "
            "Expert in perpetual swap mechanics en funding dynamiek. "
            "Bekend om het bouwen van consistente trading operaties met juiste risicoschaling."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_cro_agent() -> Agent:
    """Maak Agent 35: CRO Futures / Chief Risk Officer.

    Onafhankelijke risico-eigenaar met vetobevoegdheid en kill-switch autoriteit.
    Rapporteert aan: CEO (hiërarchisch), Groep CRO (functioneel)
    Gebruikt heavy LLM voor complexe risicobeslissingen.
    """
    tools = [
        RiskDashboardTool(),
        AlertSystemTool(),
    ] + get_futures_risk_tools()

    return create_heavy_agent(
        role="CRO Futures / Chief Risk Officer — Futures Desk Leadership (Veto/Kill-Switch)",
        goal=(
            "Onafhankelijke risico-eigenaar met veto en kill-switch autoriteit voor futures. "
            "Ontwerp risicoraamwerk: leverage caps, margin buffers, funding blootstellingslimieten. "
            "Real-time monitoring van margin en liquidatierisico. "
            "Goedkeuring nieuwe futures strategieën met leverage overwegingen. "
            "Sta tactische leverage verhogingen toe voor high-conviction setups binnen limieten."
        ),
        backstory=(
            "Risicomanagement expert met diepe derivatives ervaring. "
            "Sterk in margin risico, liquidatiemechanismen en funding blootstelling. "
            "Begrijpt leverage als een tool die strikte controles vereist. "
            "Bekend om het mogelijk maken van agressieve trading met juiste waarborgen."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_futures_coo_agent() -> Agent:
    """Maak Agent 36: COO Futures.

    Run-the-business: processen, margin management, controles.
    Rapporteert aan: CEO (hiërarchisch), Groep COO (functioneel)
    Gebruikt light LLM voor operationele taken.
    """
    tools = [
        AlertSystemTool(),
    ] + get_futures_leadership_tools()

    return create_light_agent(
        role="COO Futures — Futures Desk Operations Leadership",
        goal=(
            "Run-the-business futures: processen, margin management, controles. "
            "Stel margin monitoring, funding reconciliatie en incident runbooks in. "
            "Beheer operationele SLA's met exchanges. "
            "Handhaaf audit trail en functiescheiding. "
            "Zorg voor 24/7 dekking voor margin en liquidatie gebeurtenissen."
        ),
        backstory=(
            "Operations lead met derivatives achtergrond. "
            "Sterk in margin operaties, settlement en funding flows. "
            "Expert in het bouwen van robuuste controles voor leveraged trading."
        ),
        tools=tools,
        allow_delegation=False,
    )
