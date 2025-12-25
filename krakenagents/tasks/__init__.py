"""Tasks module for QRI Trading Organization.

Contains task factories for all agents in the organization.
"""

from krakenagents.tasks.base import create_task
from krakenagents.tasks.staff_tasks import get_staff_tasks
from krakenagents.tasks.spot_tasks import get_spot_tasks
from krakenagents.tasks.futures_tasks import get_futures_tasks

__all__ = [
    "create_task",
    "get_staff_tasks",
    "get_spot_tasks",
    "get_futures_tasks",
]
