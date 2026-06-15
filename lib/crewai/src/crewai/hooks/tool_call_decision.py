from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any


class ToolCallDecisionType(str, Enum):
    """Runtime release-control decision for a candidate tool call."""

    PROCEED = "PROCEED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    SILENCE = "SILENCE"


@dataclass(frozen=True)
class ToolCallDecision:
    """Structured release-control decision returned by before-tool hooks.

    PROCEED allows the tool call to execute. NEEDS_REVIEW and SILENCE stop
    execution before the tool runs and return a structured mediation message to
    the agent loop. Hooks may include reason and review_context metadata so the
    runtime branch is not just a bare flag.
    """

    decision: ToolCallDecisionType
    reason: str | None = None
    review_context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def proceed(
        cls,
        reason: str | None = None,
        review_context: dict[str, Any] | None = None,
    ) -> ToolCallDecision:
        """Allow the tool call to execute."""
        return cls(
            decision=ToolCallDecisionType.PROCEED,
            reason=reason,
            review_context=review_context or {},
        )

    @classmethod
    def needs_review(
        cls,
        reason: str | None = None,
        review_context: dict[str, Any] | None = None,
    ) -> ToolCallDecision:
        """Pause execution and surface review context to the agent loop."""
        return cls(
            decision=ToolCallDecisionType.NEEDS_REVIEW,
            reason=reason,
            review_context=review_context or {},
        )

    @classmethod
    def silence(
        cls,
        reason: str | None = None,
        review_context: dict[str, Any] | None = None,
    ) -> ToolCallDecision:
        """Suppress execution without treating the tool as available to run."""
        return cls(
            decision=ToolCallDecisionType.SILENCE,
            reason=reason,
            review_context=review_context or {},
        )

    @property
    def should_execute(self) -> bool:
        """Whether the runtime should execute the candidate tool call."""
        return self.decision is ToolCallDecisionType.PROCEED

    def block_message(self, tool_name: str) -> str:
        """Build the agent-visible message when execution is not released."""
        if self.decision is ToolCallDecisionType.PROCEED:
            return ""

        message = (
            "Tool execution requires review."
            if self.decision is ToolCallDecisionType.NEEDS_REVIEW
            else "Tool execution silenced by release-control hook."
        )
        details = [message, f"Tool: {tool_name}", f"Decision: {self.decision.value}"]

        if self.reason:
            details.append(f"Reason: {self.reason}")
        if self.review_context:
            details.append(
                "Review context: "
                + json.dumps(self.review_context, sort_keys=True, default=str)
            )

        return " | ".join(details)
