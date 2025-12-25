"""Futures Desk Crew for QRI Trading Organization."""

from crewai import Crew, Process

from krakenagents.agents.futures import (
    create_futures_cio_agent,
    create_futures_head_trading_agent,
    create_futures_cro_agent,
    create_futures_coo_agent,
    create_futures_systematic_head_agent,
    create_futures_systematic_operator_agent,
    create_futures_carry_head_agent,
    create_futures_carry_trader_i_agent,
    create_futures_carry_trader_ii_agent,
    create_futures_microstructure_head_agent,
    create_futures_intraday_i_agent,
    create_futures_intraday_ii_agent,
    create_futures_orderflow_agent,
    create_futures_research_head_agent,
    create_futures_funding_analyst_agent,
    create_futures_basis_analyst_agent,
    create_futures_quant_analyst_agent,
    create_futures_macro_analyst_agent,
    create_futures_flow_analyst_agent,
    create_futures_execution_head_agent,
    create_futures_unwind_specialist_agent,
    create_futures_swing_head_agent,
    create_futures_swing_btc_agent,
    create_futures_curve_trader_agent,
    create_futures_risk_monitor_agent,
    create_futures_margin_analyst_agent,
    create_futures_liquidation_agent,
    create_futures_controller_agent,
    create_futures_treasury_agent,
    create_futures_security_agent,
    create_futures_compliance_agent,
    create_futures_ops_agent,
)
from krakenagents.tasks.futures_tasks import get_futures_tasks
from krakenagents.config import get_settings, get_chat_llm


def create_futures_crew() -> Crew:
    """Create the Futures trading desk crew.

    Hierarchical crew with CIO Futures as manager.
    Handles all futures/derivatives trading operations.

    Returns:
        Configured Crew instance.
    """
    settings = get_settings()

    # Create all Futures agents
    cio = create_futures_cio_agent()
    head_trading = create_futures_head_trading_agent()
    cro = create_futures_cro_agent()
    coo = create_futures_coo_agent()

    systematic_head = create_futures_systematic_head_agent()
    systematic_operator = create_futures_systematic_operator_agent()

    carry_head = create_futures_carry_head_agent()
    carry_i = create_futures_carry_trader_i_agent()
    carry_ii = create_futures_carry_trader_ii_agent()

    micro_head = create_futures_microstructure_head_agent()
    intraday_i = create_futures_intraday_i_agent()
    intraday_ii = create_futures_intraday_ii_agent()
    orderflow = create_futures_orderflow_agent()

    research_head = create_futures_research_head_agent()
    funding = create_futures_funding_analyst_agent()
    basis = create_futures_basis_analyst_agent()
    quant = create_futures_quant_analyst_agent()
    macro = create_futures_macro_analyst_agent()
    flow = create_futures_flow_analyst_agent()

    execution_head = create_futures_execution_head_agent()
    unwind = create_futures_unwind_specialist_agent()

    swing_head = create_futures_swing_head_agent()
    swing_btc = create_futures_swing_btc_agent()
    curve = create_futures_curve_trader_agent()

    risk_monitor = create_futures_risk_monitor_agent()
    margin_analyst = create_futures_margin_analyst_agent()
    liquidation = create_futures_liquidation_agent()

    controller = create_futures_controller_agent()
    treasury = create_futures_treasury_agent()
    security = create_futures_security_agent()
    compliance = create_futures_compliance_agent()
    ops = create_futures_ops_agent()

    # Note: manager_agent (cio) should NOT be in agents list for hierarchical process
    agents = [
        # Leadership (without cio - he's the manager)
        head_trading, cro, coo,
        # Systematic
        systematic_head, systematic_operator,
        # Carry
        carry_head, carry_i, carry_ii,
        # Microstructure
        micro_head, intraday_i, intraday_ii, orderflow,
        # Research
        research_head, funding, basis, quant, macro, flow,
        # Execution
        execution_head, unwind,
        # Swing
        swing_head, swing_btc, curve,
        # Risk
        risk_monitor, margin_analyst, liquidation,
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
        "carry_head": carry_head,
        "microstructure_head": micro_head,
        "research_head": research_head,
        "execution_head": execution_head,
        "unwind_specialist": unwind,
        "swing_head": swing_head,
        "risk_monitor": risk_monitor,
        "margin_analyst": margin_analyst,
        "liquidation_specialist": liquidation,
    }
    tasks = get_futures_tasks(agents_dict)

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
