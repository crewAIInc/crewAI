"""Crews module for QRI Trading Organization.

Contains crew definitions for the trading organization:
- StaffCrew: Group Executive Board
- SpotCrew: Spot trading desk
- FuturesCrew: Futures trading desk
- TradingOrganization: Master crew coordinating all desks
"""

from krakenagents.crews.staff_crew import create_staff_crew
from krakenagents.crews.spot_crew import create_spot_crew
from krakenagents.crews.futures_crew import create_futures_crew
from krakenagents.crews.trading_org import create_trading_organization

__all__ = [
    "create_staff_crew",
    "create_spot_crew",
    "create_futures_crew",
    "create_trading_organization",
]
