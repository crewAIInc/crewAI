"""Example demonstrating select_jev_route with a Flow @router and a synthetic decision."""

from __future__ import annotations

from typing import Any

from crewai.flow.dsl import listen, router, start
from crewai.flow.flow import Flow

from crewai_tools import select_jev_route


class SupportTicketRoutingFlow(Flow):
    """Demonstrates confidence-aware Flow routing using a synthetic Jev Choice decision."""

    @start()
    def evaluate_ticket(self) -> dict[str, Any]:
        """Generate a synthetic Jev Choice decision for the incoming ticket."""
        decision = {
            "selected_choice": "REFUND_REQUEST",
            "confidence": 0.88,
        }
        self.state["decision"] = decision
        return decision

    @router(evaluate_ticket)
    def route_ticket(self, decision: dict[str, Any] | None = None) -> str:
        """Route ticket based on Jev Choice confidence thresholding."""
        ticket_decision = decision or self.state.get("decision", {})
        selected_choice = ticket_decision.get("selected_choice")
        confidence = ticket_decision.get("confidence")

        return select_jev_route(
            choice=selected_choice,
            confidence=confidence,
            allowed_routes=["REFUND_REQUEST", "TECH_SUPPORT", "GENERAL_INQUIRY"],
            min_confidence=0.80,
            fallback="GENERAL_INQUIRY",
        )

    @listen("REFUND_REQUEST")
    def process_refund(self) -> str:
        """Process refund request for high-confidence automated refund decisions."""
        return "Processed via automated refund pathway"

    @listen("TECH_SUPPORT")
    def process_tech_support(self) -> str:
        """Process technical support ticket for technical inquiry decisions."""
        return "Routed to tier-2 engineering"

    @listen("GENERAL_INQUIRY")
    def process_general_inquiry(self) -> str:
        """Handle fallback route for low-confidence or ambiguous classifications."""
        return "Routed to general triage queue (fallback)"


if __name__ == "__main__":
    flow = SupportTicketRoutingFlow()
    result = flow.kickoff()
    print("Flow completed with result:", result)  # noqa: T201
