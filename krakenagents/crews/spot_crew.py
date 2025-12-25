"""Spot Desk Crew for QRI Trading Organization."""

from crewai import Crew, Process

from krakenagents.agents.spot import (
    create_spot_cio_agent,
    create_spot_head_trading_agent,
    create_spot_cro_agent,
    create_spot_coo_agent,
    create_spot_systematic_head_agent,
    create_spot_systematic_operator_agent,
    create_spot_discretionary_head_agent,
    create_spot_swing_majors_agent,
    create_spot_swing_alts_agent,
    create_spot_arb_head_agent,
    create_spot_arb_cross_exchange_agent,
    create_spot_arb_triangular_agent,
    create_spot_research_head_agent,
    create_spot_onchain_analyst_agent,
    create_spot_tokenomics_analyst_agent,
    create_spot_data_analyst_agent,
    create_spot_macro_regime_analyst_agent,
    create_spot_news_sentiment_analyst_agent,
    create_spot_execution_head_agent,
    create_spot_intraday_btc_agent,
    create_spot_intraday_majors_agent,
    create_spot_event_trader_agent,
    create_spot_mm_supervisor_agent,
    create_spot_inventory_coordinator_agent,
    create_spot_risk_monitor_agent,
    create_spot_limits_officer_agent,
    create_spot_margin_analyst_agent,
    create_spot_controller_agent,
    create_spot_treasury_agent,
    create_spot_security_agent,
    create_spot_compliance_agent,
    create_spot_ops_agent,
)
from krakenagents.tasks.spot_tasks import get_spot_tasks
from krakenagents.config import get_settings, get_chat_llm


def create_spot_crew() -> Crew:
    """Create the Spot trading desk crew.

    Hierarchical crew with CIO Spot as manager.
    Handles all spot trading operations.

    Returns:
        Configured Crew instance.
    """
    settings = get_settings()

    # Create all Spot agents
    cio = create_spot_cio_agent()
    head_trading = create_spot_head_trading_agent()
    cro = create_spot_cro_agent()
    coo = create_spot_coo_agent()

    systematic_head = create_spot_systematic_head_agent()
    systematic_operator = create_spot_systematic_operator_agent()

    discretionary_head = create_spot_discretionary_head_agent()
    swing_majors = create_spot_swing_majors_agent()
    swing_alts = create_spot_swing_alts_agent()

    arb_head = create_spot_arb_head_agent()
    arb_cross = create_spot_arb_cross_exchange_agent()
    arb_triangular = create_spot_arb_triangular_agent()

    research_head = create_spot_research_head_agent()
    onchain = create_spot_onchain_analyst_agent()
    tokenomics = create_spot_tokenomics_analyst_agent()
    data_analyst = create_spot_data_analyst_agent()
    macro = create_spot_macro_regime_analyst_agent()
    sentiment = create_spot_news_sentiment_analyst_agent()

    execution_head = create_spot_execution_head_agent()
    intraday_btc = create_spot_intraday_btc_agent()
    intraday_majors = create_spot_intraday_majors_agent()

    event_trader = create_spot_event_trader_agent()
    mm_supervisor = create_spot_mm_supervisor_agent()

    inventory_coord = create_spot_inventory_coordinator_agent()
    risk_monitor = create_spot_risk_monitor_agent()
    limits_officer = create_spot_limits_officer_agent()
    margin_analyst = create_spot_margin_analyst_agent()

    controller = create_spot_controller_agent()
    treasury = create_spot_treasury_agent()
    security = create_spot_security_agent()
    compliance = create_spot_compliance_agent()
    ops = create_spot_ops_agent()

    # Note: manager_agent (cio) should NOT be in agents list for hierarchical process
    agents = [
        # Leadership (without cio - he's the manager)
        head_trading, cro, coo,
        # Systematic
        systematic_head, systematic_operator,
        # Discretionary
        discretionary_head, swing_majors, swing_alts,
        # Arbitrage
        arb_head, arb_cross, arb_triangular,
        # Research
        research_head, onchain, tokenomics, data_analyst, macro, sentiment,
        # Execution
        execution_head, intraday_btc, intraday_majors,
        # Event
        event_trader,
        # Market Making
        mm_supervisor,
        # Risk
        inventory_coord, risk_monitor, limits_officer, margin_analyst,
        # Operations
        controller, treasury, security, compliance, ops,
    ]

    # Create tasks
    agents_dict = {
        "cio": cio,
        "head_trading": head_trading,
        "cro": cro,
        "coo": coo,
        "systematic_head": systematic_head,
        "systematic_operator": systematic_operator,
        "discretionary_head": discretionary_head,
        "arb_head": arb_head,
        "research_head": research_head,
        "execution_head": execution_head,
        "event_trader": event_trader,
        "mm_supervisor": mm_supervisor,
        "inventory_coordinator": inventory_coord,
    }
    tasks = get_spot_tasks(agents_dict)

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.hierarchical,
        manager_agent=cio,
        chat_llm=get_chat_llm(),
        verbose=settings.crew_verbose,
        memory=settings.crew_memory,
        max_rpm=settings.crew_max_rpm,
        stream=True,
    )
