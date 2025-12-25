"""STAFF agents (STAFF-00 to STAFF-09) for QRI Trading Organization.

These are the Group Executive Board level agents that oversee
the entire trading organization.
"""

from krakenagents.agents.staff.ceo import create_ceo_agent
from krakenagents.agents.staff.group_cio import create_group_cio_agent
from krakenagents.agents.staff.group_cro import create_group_cro_agent
from krakenagents.agents.staff.group_coo import create_group_coo_agent
from krakenagents.agents.staff.group_cfo import create_group_cfo_agent
from krakenagents.agents.staff.compliance import create_compliance_agent
from krakenagents.agents.staff.security import create_security_agent
from krakenagents.agents.staff.prime import create_prime_agent
from krakenagents.agents.staff.data import create_data_agent
from krakenagents.agents.staff.people import create_people_agent

__all__ = [
    "create_ceo_agent",
    "create_group_cio_agent",
    "create_group_cro_agent",
    "create_group_coo_agent",
    "create_group_cfo_agent",
    "create_compliance_agent",
    "create_security_agent",
    "create_prime_agent",
    "create_data_agent",
    "create_people_agent",
]


def get_all_staff_agents() -> list:
    """Create and return all STAFF agents."""
    return [
        create_ceo_agent(),
        create_group_cio_agent(),
        create_group_cro_agent(),
        create_group_coo_agent(),
        create_group_cfo_agent(),
        create_compliance_agent(),
        create_security_agent(),
        create_prime_agent(),
        create_data_agent(),
        create_people_agent(),
    ]
