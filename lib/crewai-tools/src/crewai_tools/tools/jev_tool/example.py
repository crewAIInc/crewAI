"""Example demonstrating select_jev_route with a Flow @router and a synthetic decision."""

from __future__ import annotations

from typing import Any

from crewai.flow.dsl import listen, router, start
from crewai.flow.flow import Flow

from crewai_tools.tools.jev_tool.route_selector import select_jev_route


class SupportTicketRoutingFlow(Flow):
    """Demonstrates confidence-aware Flow routing using a synthetic Jev Choice decision."""

    @start()
    def evaluate_ticket(self) -> dict[str, Any]:
        # Synthetic Jev Choice answer (no live API call or credentials required)
        return {
            "selected_choice": "REFUND_REQUEST",
            "confidence": 0.88,
        }

    @router(evaluate_ticket)
    def route_ticket(self) -> str:
        # Extract decision from previous step output in state
        decision = self.state.get("evaluate_ticket", {})
        selected_choice = decision.get("selected_choice")
        confidence = decision.get("confidence")

        # Map to flow route labels using confidence threshold
        return select_jev_route(
            choice=selected_choice,
            confidence=confidence,
            allowed_routes=["REFUND_REQUEST", "TECH_SUPPORT", "GENERAL_INQUIRY"],
            min_confidence=0.80,
            fallback="GENERAL_INQUIRY",
        )

    @listen("REFUND_REQUEST")
    def process_refund(self) -> str:
        return "Processed via automated refund pathway"

    @listen("TECH_SUPPORT")
    def process_tech_support(self) -> str:
        return "Routed to tier-2 engineering"

    @listen("GENERAL_INQUIRY")
    def process_general_inquiry(self) -> str:
        return "Routed to general triage queue (fallback)"


if __name__ == "__main__":
    flow = SupportTicketRoutingFlow()
    result = flow.kickoff()
    print("Flow completed with result:", result)  # noqa: T201
