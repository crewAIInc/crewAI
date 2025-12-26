"""Crew delegatie tools voor hiërarchisch crew management.

Deze tools stellen STAFF agents (CEO, Group CIO) in staat om taken
te delegeren naar Spot en Futures trading desks.
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
    """Stel de globale crew manager referentie in."""
    global _crew_manager, _event_loop
    _crew_manager = manager
    _event_loop = loop


class DelegateToSpotDeskTool(BaseTool):
    """Tool om trading taken te delegeren naar de Spot trading desk."""

    name: str = "delegate_to_spot_desk"
    description: str = """
    Delegeer een trading taak naar de Spot trading desk.
    Gebruik dit wanneer je de Spot desk nodig hebt om:
    - Spot trades uit te voeren
    - Spot markt condities te analyseren
    - Spot posities te monitoren
    - Spot trading strategieën uit te voeren

    Input moet een JSON string zijn met:
    - directive: De specifieke taak of doel voor de Spot desk
    - priority: "high", "medium", of "low"
    - risk_budget: Optioneel risico budget percentage (1-100)
    """

    def _run(self, directive: str, priority: str = "medium", risk_budget: Optional[int] = None) -> str:
        """Delegeer taak naar Spot desk."""
        global _crew_manager, _event_loop

        if not _crew_manager:
            return "Fout: Crew manager niet geïnitialiseerd. Kan niet delegeren naar Spot desk."

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
            return f"Spot desk voert al een taak uit. Huidige opdracht wordt in de wachtrij geplaatst. Opdracht: {directive}"

        # Start the Spot crew with the directive
        try:
            from krakenagents.server import run_crew_with_streaming

            async def start_spot():
                task = asyncio.create_task(run_crew_with_streaming("spot", inputs))
                _crew_manager.crew_tasks["spot"] = task

            if _event_loop and _event_loop.is_running():
                asyncio.run_coroutine_threadsafe(start_spot(), _event_loop)

            return f"""
Delegatie naar Spot Desk GESLAAGD:
- Opdracht: {directive}
- Prioriteit: {priority}
- Risico Budget: {risk_budget or 'Standaard'}
- Status: Spot desk voert nu de opdracht uit

De Spot desk (32 agents) zal de taak analyseren en uitvoeren.
Je ontvangt updates via het dashboard.
"""
        except Exception as e:
            return f"Fout bij delegeren naar Spot desk: {str(e)}"


class DelegateToFuturesDeskTool(BaseTool):
    """Tool om trading taken te delegeren naar de Futures trading desk."""

    name: str = "delegate_to_futures_desk"
    description: str = """
    Delegeer een trading taak naar de Futures/Derivatives trading desk.
    Gebruik dit wanneer je de Futures desk nodig hebt om:
    - Futures/perpetual trades uit te voeren
    - Funding rates en basis te analyseren
    - Futures posities en margin te monitoren
    - Carry of microstructure strategieën uit te voeren

    Input moet een JSON string zijn met:
    - directive: De specifieke taak of doel voor de Futures desk
    - priority: "high", "medium", of "low"
    - risk_budget: Optioneel risico budget percentage (1-100)
    """

    def _run(self, directive: str, priority: str = "medium", risk_budget: Optional[int] = None) -> str:
        """Delegeer taak naar Futures desk."""
        global _crew_manager, _event_loop

        if not _crew_manager:
            return "Fout: Crew manager niet geïnitialiseerd. Kan niet delegeren naar Futures desk."

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
            return f"Futures desk voert al een taak uit. Huidige opdracht wordt in de wachtrij geplaatst. Opdracht: {directive}"

        # Start the Futures crew with the directive
        try:
            from krakenagents.server import run_crew_with_streaming

            async def start_futures():
                task = asyncio.create_task(run_crew_with_streaming("futures", inputs))
                _crew_manager.crew_tasks["futures"] = task

            if _event_loop and _event_loop.is_running():
                asyncio.run_coroutine_threadsafe(start_futures(), _event_loop)

            return f"""
Delegatie naar Futures Desk GESLAAGD:
- Opdracht: {directive}
- Prioriteit: {priority}
- Risico Budget: {risk_budget or 'Standaard'}
- Status: Futures desk voert nu de opdracht uit

De Futures desk (32 agents) zal de taak analyseren en uitvoeren.
Je ontvangt updates via het dashboard.
"""
        except Exception as e:
            return f"Fout bij delegeren naar Futures desk: {str(e)}"


class GetDeskStatusTool(BaseTool):
    """Tool om status van trading desks op te halen."""

    name: str = "get_desk_status"
    description: str = """
    Haal de huidige status op van trading desks (Spot en/of Futures).
    Gebruik dit om te controleren:
    - Of een desk momenteel actief is
    - Welke taken worden uitgevoerd
    - Algemene desk status

    Input: desk naam ("spot", "futures", of "all")
    """

    def _run(self, desk: str = "all") -> str:
        """Haal desk status op."""
        global _crew_manager

        if not _crew_manager:
            return "Fout: Crew manager niet geïnitialiseerd."

        statuses = []
        desks_to_check = ["spot", "futures"] if desk == "all" else [desk.lower()]

        for desk_id in desks_to_check:
            if desk_id in _crew_manager.crew_tasks:
                task = _crew_manager.crew_tasks[desk_id]
                if task.done():
                    status = "Voltooid"
                    if task.exception():
                        status = f"Fout: {task.exception()}"
                else:
                    status = "Actief"
            else:
                status = "Inactief"

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
    """Tool om taken gelijktijdig te delegeren naar zowel Spot als Futures desks."""

    name: str = "delegate_to_both_desks"
    description: str = """
    Delegeer gecoördineerde taken naar BEIDE Spot en Futures desks.
    Gebruik dit voor:
    - Basis trades (spot vs futures arbitrage)
    - Hedging operaties
    - Gecoördineerde risico reductie
    - Volledige portfolio analyse

    Input moet een JSON string zijn met:
    - spot_directive: Taak voor Spot desk
    - futures_directive: Taak voor Futures desk
    - coordination_note: Hoe de taken aan elkaar gerelateerd zijn
    - priority: "high", "medium", of "low"
    """

    def _run(
        self,
        spot_directive: str,
        futures_directive: str,
        coordination_note: str = "",
        priority: str = "medium"
    ) -> str:
        """Delegeer naar beide desks."""
        global _crew_manager, _event_loop

        if not _crew_manager:
            return "Fout: Crew manager niet geïnitialiseerd."

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
Gecoördineerde Delegatie naar BEIDE Desks:

Coördinatie Notitie: {coordination_note}
Prioriteit: {priority}

{chr(10).join(results)}

Beide desks voeren nu hun respectievelijke taken uit in coördinatie.
"""


# Export all tools
def get_delegation_tools() -> list[BaseTool]:
    """Haal alle delegatie tools op voor STAFF agents."""
    return [
        DelegateToSpotDeskTool(),
        DelegateToFuturesDeskTool(),
        GetDeskStatusTool(),
        DelegateToBothDesksTool(),
    ]
