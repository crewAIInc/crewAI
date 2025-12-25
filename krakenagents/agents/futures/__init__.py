"""Futures/Derivatives Desk agents (Agent 33-64) for QRI Trading Organization.

These are the Futures/Derivatives trading desk agents organized by function:
- Leadership (33-36): CIO, Head of Trading, CRO, COO
- Systematic (37, 42): Systematic trading
- Carry (38, 43, 44): Funding/carry trading
- Microstructure (39, 45-47): Intraday/orderflow trading
- Research (40, 52-56): Research and analysis
- Execution (41, 51): Order execution and unwind
- Swing (48-50): Swing and curve trading
- Risk (57-59): Risk and margin management
- Operations (60-64): Controller, Treasury, Security, Compliance, Ops
"""

from krakenagents.agents.futures.leadership import (
    create_futures_cio_agent,
    create_futures_head_trading_agent,
    create_futures_cro_agent,
    create_futures_coo_agent,
)
from krakenagents.agents.futures.systematic import (
    create_futures_systematic_head_agent,
    create_futures_systematic_operator_agent,
)
from krakenagents.agents.futures.carry import (
    create_futures_carry_head_agent,
    create_futures_carry_trader_i_agent,
    create_futures_carry_trader_ii_agent,
)
from krakenagents.agents.futures.microstructure import (
    create_futures_microstructure_head_agent,
    create_futures_intraday_i_agent,
    create_futures_intraday_ii_agent,
    create_futures_orderflow_agent,
)
from krakenagents.agents.futures.research import (
    create_futures_research_head_agent,
    create_futures_funding_analyst_agent,
    create_futures_basis_analyst_agent,
    create_futures_quant_analyst_agent,
    create_futures_macro_analyst_agent,
    create_futures_flow_analyst_agent,
)
from krakenagents.agents.futures.execution import (
    create_futures_execution_head_agent,
    create_futures_unwind_specialist_agent,
)
from krakenagents.agents.futures.swing import (
    create_futures_swing_head_agent,
    create_futures_swing_btc_agent,
    create_futures_curve_trader_agent,
)
from krakenagents.agents.futures.risk import (
    create_futures_risk_monitor_agent,
    create_futures_margin_analyst_agent,
    create_futures_liquidation_agent,
)
from krakenagents.agents.futures.operations import (
    create_futures_controller_agent,
    create_futures_treasury_agent,
    create_futures_security_agent,
    create_futures_compliance_agent,
    create_futures_ops_agent,
)

__all__ = [
    # Leadership
    "create_futures_cio_agent",
    "create_futures_head_trading_agent",
    "create_futures_cro_agent",
    "create_futures_coo_agent",
    # Systematic
    "create_futures_systematic_head_agent",
    "create_futures_systematic_operator_agent",
    # Carry
    "create_futures_carry_head_agent",
    "create_futures_carry_trader_i_agent",
    "create_futures_carry_trader_ii_agent",
    # Microstructure
    "create_futures_microstructure_head_agent",
    "create_futures_intraday_i_agent",
    "create_futures_intraday_ii_agent",
    "create_futures_orderflow_agent",
    # Research
    "create_futures_research_head_agent",
    "create_futures_funding_analyst_agent",
    "create_futures_basis_analyst_agent",
    "create_futures_quant_analyst_agent",
    "create_futures_macro_analyst_agent",
    "create_futures_flow_analyst_agent",
    # Execution
    "create_futures_execution_head_agent",
    "create_futures_unwind_specialist_agent",
    # Swing
    "create_futures_swing_head_agent",
    "create_futures_swing_btc_agent",
    "create_futures_curve_trader_agent",
    # Risk
    "create_futures_risk_monitor_agent",
    "create_futures_margin_analyst_agent",
    "create_futures_liquidation_agent",
    # Operations
    "create_futures_controller_agent",
    "create_futures_treasury_agent",
    "create_futures_security_agent",
    "create_futures_compliance_agent",
    "create_futures_ops_agent",
]


def get_all_futures_agents() -> list:
    """Create and return all Futures desk agents (32 agents)."""
    return [
        # Leadership (33-36)
        create_futures_cio_agent(),
        create_futures_head_trading_agent(),
        create_futures_cro_agent(),
        create_futures_coo_agent(),
        # Systematic (37, 42)
        create_futures_systematic_head_agent(),
        create_futures_systematic_operator_agent(),
        # Carry (38, 43, 44)
        create_futures_carry_head_agent(),
        create_futures_carry_trader_i_agent(),
        create_futures_carry_trader_ii_agent(),
        # Microstructure (39, 45-47)
        create_futures_microstructure_head_agent(),
        create_futures_intraday_i_agent(),
        create_futures_intraday_ii_agent(),
        create_futures_orderflow_agent(),
        # Research (40, 52-56)
        create_futures_research_head_agent(),
        create_futures_funding_analyst_agent(),
        create_futures_basis_analyst_agent(),
        create_futures_quant_analyst_agent(),
        create_futures_macro_analyst_agent(),
        create_futures_flow_analyst_agent(),
        # Execution (41, 51)
        create_futures_execution_head_agent(),
        create_futures_unwind_specialist_agent(),
        # Swing (48-50)
        create_futures_swing_head_agent(),
        create_futures_swing_btc_agent(),
        create_futures_curve_trader_agent(),
        # Risk (57-59)
        create_futures_risk_monitor_agent(),
        create_futures_margin_analyst_agent(),
        create_futures_liquidation_agent(),
        # Operations (60-64)
        create_futures_controller_agent(),
        create_futures_treasury_agent(),
        create_futures_security_agent(),
        create_futures_compliance_agent(),
        create_futures_ops_agent(),
    ]
