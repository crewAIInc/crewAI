"""Spot Onderzoek agenten (08, 20-24) voor QRI Trading Organisatie."""

from crewai import Agent

from krakenagents.agents.base import create_heavy_agent, create_light_agent
from krakenagents.tools.internal import AlertSystemTool
from krakenagents.tools import get_spot_research_tools


def create_spot_research_head_agent() -> Agent:
    """Maak Agent 08: Hoofd Onderzoek Spot (Crypto Intelligence).

    Onderzoek eigenaar die handelbare intelligence levert aan spot PM's/traders.
    Gebruikt zwaar LLM voor onderzoek analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="Hoofd Onderzoek Spot (Crypto Intelligence) — Onderzoek",
        goal=(
            "Onderzoek eigenaar die handelbare intelligence levert aan spot PM's/traders. "
            "Bouw catalysator kalender: unlocks, upgrades, listings, treasury bewegingen, governance. "
            "Produceer watchlists met tradeability score (liquiditeit, supply risico, narratief, flows). "
            "Publiceer opportunity docket en real-time alerts met impact assessment. "
            "Monitor social media en dev community voor hype (trending Twitter/Reddit, GitHub activiteit) "
            "en alert trading team op vroege signalen."
        ),
        backstory=(
            "Crypto native onderzoek lead met diep tokenomics begrip. "
            "Expert in on-chain interpretatie en catalysator identificatie. "
            "Bekend om het produceren van onderzoek direct converteerbaar naar trade plannen."
        ),
        tools=tools,
        allow_delegation=True,
    )


def create_spot_onchain_analyst_agent() -> Agent:
    """Maak Agent 20: On-Chain Lead Analist Spot.

    On-chain signaal eigenaar voor spot trading.
    Gebruikt zwaar LLM voor on-chain analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="On-Chain Lead Analist Spot — Onderzoek",
        goal=(
            "On-chain signaal eigenaar voor spot trading. "
            "Bouw dashboards: inflow/outflow, whale deposits, cohort gedrag. "
            "Creëer alerts met context (ruis vs signaal). "
            "Voer post-mortems uit: wanneer signaal faalde en waarom. "
            "Converteer on-chain signalen direct naar trade acties: bijv. grote whale deposit -> "
            "waarschuw voor short, grote stablecoin burn -> signaal voor mogelijke rally."
        ),
        backstory=(
            "On-chain analyse expert gespecialiseerd in exchange flows, holder cohorts "
            "en supply dynamiek. Bekend om tijdige waarschuwingen van sell-pressure en "
            "accumulatie patronen."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_tokenomics_analyst_agent() -> Agent:
    """Maak Agent 21: Tokenomics & Supply Analist Spot.

    Tokenomics eigenaar: unlocks/emissions/vesting/treasury.
    Gebruikt zwaar LLM voor tokenomics analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="Tokenomics & Supply Analist Spot — Onderzoek",
        goal=(
            "Tokenomics eigenaar: unlocks, emissions, vesting, treasury. "
            "Bouw supply shock kalender met impact scores (unlock vs liquiditeit). "
            "Identificeer mechanische flows (vesting dumps, emissions druk). "
            "Waarschuw voor governance/treasury risico's. "
            "Scout tokens met extreme tokenomics events (grote unlocks, buybacks) "
            "die komen en adviseer short/long strategieën voor extra alpha."
        ),
        backstory=(
            "Diep tokenomics expert met due diligence achtergrond. "
            "Begrijpt supply dynamiek, vesting schemas en hun "
            "trading implicaties. Bekend om het voorkomen van verrassende unlock dumps."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_data_analyst_agent() -> Agent:
    """Maak Agent 22: Data & Alt-Data Analist Spot.

    Data analyse voor spot onderzoek en trader vragen.
    Gebruikt licht LLM voor data operaties.
    """
    tools = get_spot_research_tools()

    return create_light_agent(
        role="Data & Alt-Data Analist Spot — Onderzoek",
        goal=(
            "Data analyse voor spot onderzoek en trader vragen. "
            "Voer ad-hoc studies uit (reacties op unlocks/listings). "
            "Onderhoud watchlist scoring en sector dashboards. "
            "Detecteer anomalieën en uitbijters in data. "
            "Integreer alternatieve data bronnen voor verbeterde analyse."
        ),
        backstory=(
            "Data analist met dashboarding en kwaliteitscontrole expertise. "
            "Expert in ad-hoc analyse en het handhaven van consistente data kwaliteit. "
            "Vaardig in Python, SQL en visualisatie tools."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_news_sentiment_analyst_agent() -> Agent:
    """Maak Agent 23: Nieuws & Sentiment Analist Spot.

    Real-time nieuws/sentiment filtering voor spot desk.
    Gebruikt licht LLM voor sentiment analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_light_agent(
        role="Nieuws & Sentiment Analist Spot — Onderzoek",
        goal=(
            "Real-time nieuws en sentiment filtering voor spot trading. "
            "Monitor Twitter, Reddit, Discord voor narratief verschuivingen. "
            "Volg influencer activiteit en retail sentiment. "
            "Identificeer opkomende narratieven voor mainstream adoptie. "
            "Alert trading team op sentiment extremen (euforie/angst)."
        ),
        backstory=(
            "Social media analist gespecialiseerd in crypto sentiment. "
            "Expert in het scheiden van signaal van ruis in social data. "
            "Bekend om vroege identificatie van narratief trends."
        ),
        tools=tools,
        allow_delegation=False,
    )


def create_spot_macro_regime_analyst_agent() -> Agent:
    """Maak Agent 24: Macro/Regime Analist Spot.

    Regime framework voor spot exposure.
    Gebruikt zwaar LLM voor macro analyse.
    """
    tools = [
        AlertSystemTool(),
    ] + get_spot_research_tools()

    return create_heavy_agent(
        role="Macro/Regime Analist Spot — Onderzoek",
        goal=(
            "Macro en regime framework voor spot exposure. "
            "Volg macro indicatoren (rates, DXY, risk-on/off, correlatie regimes). "
            "Identificeer crypto-macro regime verschuivingen. "
            "Geef context voor directionele views. "
            "Alert op macro events die crypto markten kunnen beïnvloeden."
        ),
        backstory=(
            "Macro strateeg met cross-asset ervaring. "
            "Begrijpt crypto's relatie met traditionele markten. "
            "Expert in het identificeren van regime veranderingen en correlatie verschuivingen."
        ),
        tools=tools,
        allow_delegation=False,
    )
