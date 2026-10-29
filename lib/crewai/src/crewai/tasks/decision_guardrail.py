"""Guardrail that approves a task output from a typed predicate decision.

The guardrail asks one predicate question about the output through a
:class:`PredicateDecider` (for example an OpenAI Decisions adapter) and passes
only when the returned probability meets a caller-configured threshold.

This module deliberately avoids ``from __future__ import annotations``: the
Task guardrail validator inspects ``__call__``'s return annotation at runtime
and requires a real ``tuple[bool, Any]``, not a string.
"""

import math
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from crewai.lite_agent_output import LiteAgentOutput
from crewai.tasks.task_output import TaskOutput


class PredicateAnswer(BaseModel):
    """A typed answer to one predicate question.

    Attributes:
        name: The question name echoed back by the decider.
        probability: Estimated probability, from 0 to 1, that the predicate holds.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    probability: float = Field(ge=0.0, le=1.0)


class PredicateDecider(Protocol):
    """What :class:`DecisionGuardrail` needs from a decisions adapter."""

    def predicate(
        self, input: str, *, name: str, instructions: str
    ) -> PredicateAnswer | None:
        """Ask one predicate question about ``input``.

        Returns:
            The answer, or None when no answer was produced for the question
            (missing or refused).

        Raises:
            Exception: On API or transport errors.
        """
        ...


def _is_unit_probability(value: Any) -> bool:
    """Whether value is a real, finite number in [0, 1] (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0.0 <= value <= 1.0


class DecisionGuardrail:
    """Pass a task output when a predicate's probability meets a threshold.

    The guardrail fails closed: a missing answer, an answer for another
    question, a malformed probability, or any decider error rejects the output
    rather than approving it. A ``HookAborted`` raised by the decider propagates.

    Args:
        instructions: The condition the output must satisfy.
        decider: The adapter that answers the predicate question.
        threshold: Minimum probability, from 0 to 1, required to pass. An answer
            exactly at the threshold passes.
        name: The question name sent to and expected back from the decider.

    Attributes:
        last_answer: The valid answer from the most recent evaluation, or None
            when the most recent evaluation produced no valid answer.

    Example:
        >>> guardrail = DecisionGuardrail(
        ...     "The summary cites at least one source",
        ...     decider=my_decisions_adapter,
        ...     threshold=0.8,
        ... )
        >>> task = Task(..., guardrail=guardrail)
    """

    def __init__(
        self,
        instructions: str,
        decider: PredicateDecider,
        *,
        threshold: float,
        name: str = "task_output_check",
    ) -> None:
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be a non-empty string")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not _is_unit_probability(threshold):
            raise ValueError(
                f"threshold must be a finite number between 0 and 1, got {threshold!r}"
            )

        self.instructions = instructions
        self.decider = decider
        self.threshold = float(threshold)
        self.name = name
        self.last_answer: PredicateAnswer | None = None

    @property
    def description(self) -> str:
        """Describe this guardrail for event logging."""
        return f"Decision predicate: {self.instructions}"

    def _invalid_answer_reason(self, answer: Any) -> str | None:
        """Explain why answer cannot be trusted, or return None if it is valid."""
        answer_name = getattr(answer, "name", None)
        if answer_name != self.name:
            return (
                f"Decision answered predicate {answer_name!r}, expected {self.name!r}."
            )
        if not _is_unit_probability(getattr(answer, "probability", None)):
            return f"Decision for predicate {self.name!r} has no valid probability."
        return None

    def __call__(self, task_output: TaskOutput | LiteAgentOutput) -> tuple[bool, Any]:
        """Validate the output against the predicate.

        Args:
            task_output: The output to validate.

        Returns:
            ``(True, task_output.raw)`` when the probability meets the
            threshold, otherwise ``(False, error_message)``.

        Raises:
            HookAborted: A hook denied the decider's model call.
        """
        from crewai.hooks.dispatch import HookAborted

        self.last_answer = None
        try:
            answer = self.decider.predicate(
                task_output.raw, name=self.name, instructions=self.instructions
            )
        except HookAborted:
            raise
        except Exception as e:
            return False, f"Decision guardrail could not evaluate the output: {e!s}"

        if answer is None:
            return False, f"No decision was returned for predicate {self.name!r}."
        reason = self._invalid_answer_reason(answer)
        if reason is not None:
            return False, reason

        self.last_answer = answer
        if answer.probability >= self.threshold:
            return True, task_output.raw
        return False, (
            f"Output did not satisfy the check {self.instructions!r} "
            f"(probability {answer.probability:.3f} < required {self.threshold:.3f})."
        )
