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

    decision: ToolCallDecisionType | str
    reason: str | None = None
    review_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "review_context",
            self._copy_context(self.review_context),
        )
        try:
            object.__setattr__(self, "decision", ToolCallDecisionType(self.decision))
        except ValueError:
            pass

    @staticmethod
    def _copy_context(review_context: dict[str, Any] | None) -> dict[str, Any]:
        return dict(review_context) if review_context is not None else {}

    @staticmethod
    def _serialize_context(review_context: dict[str, Any]) -> str:
        try:
            return json.dumps(review_context, sort_keys=True, default=str)
        except Exception:
            try:
                return repr(review_context)
            except Exception:
                return "<unserializable review_context>"

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
            review_context=cls._copy_context(review_context),
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
            review_context=cls._copy_context(review_context),
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
            review_context=cls._copy_context(review_context),
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
        decision_value = (
            self.decision.value
            if isinstance(self.decision, ToolCallDecisionType)
            else str(self.decision)
        )
        details = [message, f"Tool: {tool_name}", f"Decision: {decision_value}"]

        if self.reason:
            details.append(f"Reason: {self.reason}")
        if self.review_context:
            details.append(
                "Review context: " + self._serialize_context(self.review_context)
            )

        return " | ".join(details)
