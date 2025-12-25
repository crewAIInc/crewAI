"""Spot Desk agents (Agent 01-32) for QRI Trading Organization.

These are the Spot trading desk agents organized by function:
- Leadership (01-04): CIO, Head of Trading, CRO, COO
- Systematic (05, 10): Systematic trading
- Discretionary (06, 13, 14): Swing and thematic trading
- Arbitrage (07, 11, 12): Cross-exchange and triangular arb
- Research (08, 20-24): Research and analysis
- Execution (09, 16, 17): Order execution and intraday trading
- Event (15): Event-driven trading
- Market Making (18): Liquidity provision
- Risk (19, 25-27): Risk and inventory management
- Operations (28-32): Controller, Treasury, Security, Compliance, Ops
"""

from krakenagents.agents.spot.leadership import (
    create_spot_cio_agent,
    create_spot_head_trading_agent,
    create_spot_cro_agent,
    create_spot_coo_agent,
)
from krakenagents.agents.spot.systematic import (
    create_spot_systematic_head_agent,
    create_spot_systematic_operator_agent,
)
from krakenagents.agents.spot.discretionary import (
    create_spot_discretionary_head_agent,
    create_spot_swing_majors_agent,
    create_spot_swing_alts_agent,
)
from krakenagents.agents.spot.arbitrage import (
    create_spot_arb_head_agent,
    create_spot_arb_cross_exchange_agent,
    create_spot_arb_triangular_agent,
)
from krakenagents.agents.spot.research import (
    create_spot_research_head_agent,
    create_spot_onchain_analyst_agent,
    create_spot_tokenomics_analyst_agent,
    create_spot_data_analyst_agent,
    create_spot_news_sentiment_analyst_agent,
    create_spot_macro_regime_analyst_agent,
)
from krakenagents.agents.spot.execution import (
    create_spot_execution_head_agent,
    create_spot_intraday_btc_agent,
    create_spot_intraday_majors_agent,
)
from krakenagents.agents.spot.event import create_spot_event_trader_agent
from krakenagents.agents.spot.market_making import create_spot_mm_supervisor_agent
from krakenagents.agents.spot.risk import (
    create_spot_inventory_coordinator_agent,
    create_spot_risk_monitor_agent,
    create_spot_limits_officer_agent,
    create_spot_margin_analyst_agent,
)
from krakenagents.agents.spot.operations import (
    create_spot_controller_agent,
    create_spot_treasury_agent,
    create_spot_security_agent,
    create_spot_compliance_agent,
    create_spot_ops_agent,
)

__all__ = [
    # Leadership
    "create_spot_cio_agent",
    "create_spot_head_trading_agent",
    "create_spot_cro_agent",
    "create_spot_coo_agent",
    # Systematic
    "create_spot_systematic_head_agent",
    "create_spot_systematic_operator_agent",
    # Discretionary
    "create_spot_discretionary_head_agent",
    "create_spot_swing_majors_agent",
    "create_spot_swing_alts_agent",
    # Arbitrage
    "create_spot_arb_head_agent",
    "create_spot_arb_cross_exchange_agent",
    "create_spot_arb_triangular_agent",
    # Research
    "create_spot_research_head_agent",
    "create_spot_onchain_analyst_agent",
    "create_spot_tokenomics_analyst_agent",
    "create_spot_data_analyst_agent",
    "create_spot_news_sentiment_analyst_agent",
    "create_spot_macro_regime_analyst_agent",
    # Execution
    "create_spot_execution_head_agent",
    "create_spot_intraday_btc_agent",
    "create_spot_intraday_majors_agent",
    # Event
    "create_spot_event_trader_agent",
    # Market Making
    "create_spot_mm_supervisor_agent",
    # Risk
    "create_spot_inventory_coordinator_agent",
    "create_spot_risk_monitor_agent",
    "create_spot_limits_officer_agent",
    "create_spot_margin_analyst_agent",
    # Operations
    "create_spot_controller_agent",
    "create_spot_treasury_agent",
    "create_spot_security_agent",
    "create_spot_compliance_agent",
    "create_spot_ops_agent",
]


def get_all_spot_agents() -> list:
    """Create and return all Spot desk agents (32 agents)."""
    return [
        # Leadership (01-04)
        create_spot_cio_agent(),
        create_spot_head_trading_agent(),
        create_spot_cro_agent(),
        create_spot_coo_agent(),
        # Systematic (05, 10)
        create_spot_systematic_head_agent(),
        create_spot_systematic_operator_agent(),
        # Discretionary (06, 13, 14)
        create_spot_discretionary_head_agent(),
        create_spot_swing_majors_agent(),
        create_spot_swing_alts_agent(),
        # Arbitrage (07, 11, 12)
        create_spot_arb_head_agent(),
        create_spot_arb_cross_exchange_agent(),
        create_spot_arb_triangular_agent(),
        # Research (08, 20-24)
        create_spot_research_head_agent(),
        create_spot_onchain_analyst_agent(),
        create_spot_tokenomics_analyst_agent(),
        create_spot_data_analyst_agent(),
        create_spot_news_sentiment_analyst_agent(),
        create_spot_macro_regime_analyst_agent(),
        # Execution (09, 16, 17)
        create_spot_execution_head_agent(),
        create_spot_intraday_btc_agent(),
        create_spot_intraday_majors_agent(),
        # Event (15)
        create_spot_event_trader_agent(),
        # Market Making (18)
        create_spot_mm_supervisor_agent(),
        # Risk (19, 25-27)
        create_spot_inventory_coordinator_agent(),
        create_spot_risk_monitor_agent(),
        create_spot_limits_officer_agent(),
        create_spot_margin_analyst_agent(),
        # Operations (28-32)
        create_spot_controller_agent(),
        create_spot_treasury_agent(),
        create_spot_security_agent(),
        create_spot_compliance_agent(),
        create_spot_ops_agent(),
    ]
