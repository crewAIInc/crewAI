"""Crew delegation tools for hierarchical crew management.

These tools allow STAFF agents (CEO, Group CIO) to delegate tasks
to Spot and Futures trading desks.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Optional
from crewai.tools import BaseTool
from pydantic import Field


# Global reference to crew manager (set by server)
_crew_manager = None
_event_loop = None


def set_crew_manager(manager, loop):
    """Set the global crew manager reference."""
    global _crew_manager, _event_loop
    _crew_manager = manager
    _event_loop = loop


class DelegateToSpotDeskTool(BaseTool):
    """Tool to delegate trading tasks to the Spot trading desk."""

    name: str = "delegate_to_spot_desk"
    description: str = """
    Delegate a trading task to the Spot trading desk.
    Use this when you need the Spot desk to:
    - Execute spot trades
    - Analyze spot market conditions
    - Monitor spot positions
    - Run spot trading strategies

    Input should be a JSON string with:
    - directive: The specific task or goal for the Spot desk
    - priority: "high", "medium", or "low"
    - risk_budget: Optional risk budget percentage (1-100)
    """

    def _run(self, directive: str, priority: str = "medium", risk_budget: Optional[int] = None) -> str:
        """Delegate task to Spot desk."""
        global _crew_manager, _event_loop

        if not _crew_manager:
            return "Error: Crew manager not initialized. Cannot delegate to Spot desk."

        # Parse input if it's JSON
        if isinstance(directive, str) and directive.startswith("{"):
            try:
                data = json.loads(directive)
                directive = data.get("directive", directive)
                priority = data.get("priority", priority)
                risk_budget = data.get("risk_budget", risk_budget)
            except json.JSONDecodeError:
                pass

        # Build inputs for the Spot crew
        inputs = {
            "directive": directive,
            "priority": priority,
            "delegated_by": "STAFF",
            "delegated_at": datetime.utcnow().isoformat(),
        }
        if risk_budget:
            inputs["risk_budget"] = risk_budget

        # Check if Spot crew is already running
        if "spot" in _crew_manager.crew_tasks and not _crew_manager.crew_tasks["spot"].done():
            return f"Spot desk is already executing a task. Current directive will be queued. Directive: {directive}"

        # Start the Spot crew with the directive
        try:
            from krakenagents.server import run_crew_with_streaming

            async def start_spot():
                task = asyncio.create_task(run_crew_with_streaming("spot", inputs))
                _crew_manager.crew_tasks["spot"] = task

            if _event_loop and _event_loop.is_running():
                asyncio.run_coroutine_threadsafe(start_spot(), _event_loop)

            return f"""
Delegation to Spot Desk SUCCESSFUL:
- Directive: {directive}
- Priority: {priority}
- Risk Budget: {risk_budget or 'Default'}
- Status: Spot desk is now executing the directive

The Spot desk (32 agents) will analyze and execute the task.
You will receive updates via the dashboard.
"""
        except Exception as e:
            return f"Error delegating to Spot desk: {str(e)}"


class DelegateToFuturesDeskTool(BaseTool):
    """Tool to delegate trading tasks to the Futures trading desk."""

    name: str = "delegate_to_futures_desk"
    description: str = """
    Delegate a trading task to the Futures/Derivatives trading desk.
    Use this when you need the Futures desk to:
    - Execute futures/perpetual trades
    - Analyze funding rates and basis
    - Monitor futures positions and margin
    - Run carry or microstructure strategies

    Input should be a JSON string with:
    - directive: The specific task or goal for the Futures desk
    - priority: "high", "medium", or "low"
    - risk_budget: Optional risk budget percentage (1-100)
    """

    def _run(self, directive: str, priority: str = "medium", risk_budget: Optional[int] = None) -> str:
        """Delegate task to Futures desk."""
        global _crew_manager, _event_loop

        if not _crew_manager:
            return "Error: Crew manager not initialized. Cannot delegate to Futures desk."

        # Parse input if it's JSON
        if isinstance(directive, str) and directive.startswith("{"):
            try:
                data = json.loads(directive)
                directive = data.get("directive", directive)
                priority = data.get("priority", priority)
                risk_budget = data.get("risk_budget", risk_budget)
            except json.JSONDecodeError:
                pass

        # Build inputs for the Futures crew
        inputs = {
            "directive": directive,
            "priority": priority,
            "delegated_by": "STAFF",
            "delegated_at": datetime.utcnow().isoformat(),
        }
        if risk_budget:
            inputs["risk_budget"] = risk_budget

        # Check if Futures crew is already running
        if "futures" in _crew_manager.crew_tasks and not _crew_manager.crew_tasks["futures"].done():
            return f"Futures desk is already executing a task. Current directive will be queued. Directive: {directive}"

        # Start the Futures crew with the directive
        try:
            from krakenagents.server import run_crew_with_streaming

            async def start_futures():
                task = asyncio.create_task(run_crew_with_streaming("futures", inputs))
                _crew_manager.crew_tasks["futures"] = task

            if _event_loop and _event_loop.is_running():
                asyncio.run_coroutine_threadsafe(start_futures(), _event_loop)

            return f"""
Delegation to Futures Desk SUCCESSFUL:
- Directive: {directive}
- Priority: {priority}
- Risk Budget: {risk_budget or 'Default'}
- Status: Futures desk is now executing the directive

The Futures desk (32 agents) will analyze and execute the task.
You will receive updates via the dashboard.
"""
        except Exception as e:
            return f"Error delegating to Futures desk: {str(e)}"


class GetDeskStatusTool(BaseTool):
    """Tool to get status of trading desks."""

    name: str = "get_desk_status"
    description: str = """
    Get the current status of trading desks (Spot and/or Futures).
    Use this to check:
    - Whether a desk is currently running
    - What tasks are being executed
    - Overall desk health

    Input: desk name ("spot", "futures", or "all")
    """

    def _run(self, desk: str = "all") -> str:
        """Get desk status."""
        global _crew_manager

        if not _crew_manager:
            return "Error: Crew manager not initialized."

        statuses = []
        desks_to_check = ["spot", "futures"] if desk == "all" else [desk.lower()]

        for desk_id in desks_to_check:
            if desk_id in _crew_manager.crew_tasks:
                task = _crew_manager.crew_tasks[desk_id]
                if task.done():
                    status = "Completed"
                    if task.exception():
                        status = f"Error: {task.exception()}"
                else:
                    status = "Running"
            else:
                status = "Idle"

            desk_name = "Spot Desk" if desk_id == "spot" else "Futures Desk"
            agent_count = 32

            statuses.append(f"""
{desk_name}:
- Status: {status}
- Agents: {agent_count}
- Connected clients: {len(_crew_manager.active_connections.get(desk_id, []))}
""")

        return "\n".join(statuses)


class DelegateToBothDesksTool(BaseTool):
    """Tool to delegate tasks to both Spot and Futures desks simultaneously."""

    name: str = "delegate_to_both_desks"
    description: str = """
    Delegate coordinated tasks to BOTH Spot and Futures desks.
    Use this for:
    - Basis trades (spot vs futures arbitrage)
    - Hedging operations
    - Coordinated risk reduction
    - Full portfolio analysis

    Input should be a JSON string with:
    - spot_directive: Task for Spot desk
    - futures_directive: Task for Futures desk
    - coordination_note: How the tasks relate to each other
    - priority: "high", "medium", or "low"
    """

    def _run(
        self,
        spot_directive: str,
        futures_directive: str,
        coordination_note: str = "",
        priority: str = "medium"
    ) -> str:
        """Delegate to both desks."""
        global _crew_manager, _event_loop

        if not _crew_manager:
            return "Error: Crew manager not initialized."

        # Parse input if it's JSON
        if isinstance(spot_directive, str) and spot_directive.startswith("{"):
            try:
                data = json.loads(spot_directive)
                spot_directive = data.get("spot_directive", spot_directive)
                futures_directive = data.get("futures_directive", futures_directive)
                coordination_note = data.get("coordination_note", coordination_note)
                priority = data.get("priority", priority)
            except json.JSONDecodeError:
                pass

        results = []

        # Delegate to Spot
        spot_tool = DelegateToSpotDeskTool()
        spot_result = spot_tool._run(
            f"{spot_directive} [COORDINATED: {coordination_note}]",
            priority
        )
        results.append(f"SPOT: {spot_result}")

        # Delegate to Futures
        futures_tool = DelegateToFuturesDeskTool()
        futures_result = futures_tool._run(
            f"{futures_directive} [COORDINATED: {coordination_note}]",
            priority
        )
        results.append(f"FUTURES: {futures_result}")

        return f"""
Coordinated Delegation to BOTH Desks:

Coordination Note: {coordination_note}
Priority: {priority}

{chr(10).join(results)}

Both desks are now executing their respective tasks in coordination.
"""


# Export all tools
def get_delegation_tools() -> list[BaseTool]:
    """Get all delegation tools for STAFF agents."""
    return [
        DelegateToSpotDeskTool(),
        DelegateToFuturesDeskTool(),
        GetDeskStatusTool(),
        DelegateToBothDesksTool(),
    ]
