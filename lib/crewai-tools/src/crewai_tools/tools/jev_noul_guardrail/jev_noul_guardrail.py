"""Callable task guardrail backed by one TypeSafe Jev Noul question.

Jev answers a Noul question with the probability that the question is true. The
guardrail asks one such question about a task output and accepts the output when
that probability meets a caller-configured threshold.
"""

# No `from __future__ import annotations` here: Task validates a guardrail's return
# annotation at runtime and only accepts a real `tuple[bool, Any]`, not a string.

from collections.abc import Mapping
import importlib
import math
from typing import Any, Protocol

from crewai.lite_agent_output import LiteAgentOutput
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.guardrail_types import GuardrailCallable
from typing_extensions import TypeIs


DEFAULT_QUESTION_NAME = "output_passes"


class SystemOneClient(Protocol):
    """The part of a TypeSafe client that the guardrail uses."""

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> Any:
        """Answer named questions about ``state``."""
        ...


def create_jev_noul_guardrail(
    question: str,
    threshold: float,
    *,
    client: SystemOneClient | None = None,
    model: str | None = None,
    question_name: str = DEFAULT_QUESTION_NAME,
) -> GuardrailCallable:
    """Create a callable task guardrail that asks Jev one Noul question.

    The question is sent with the task output's raw text as the state. Phrase it so
    that "true" means the output satisfies the guardrail, for example "Does this
    answer cite at least one source?". The output passes when Jev's probability
    that the question is true is greater than or equal to ``threshold``.

    A missing, malformed, non-finite or out-of-range answer rejects the output.
    Errors raised by the TypeSafe SDK (authentication, network, API or response
    validation errors) are not caught, so they surface to the caller instead of
    being turned into a pass or a retry. Rejections use the task's existing
    guardrail retry mechanism (``guardrail_max_retries``).

    Args:
        question: The Noul question to ask about the output.
        threshold: Minimum probability, a finite number from 0 to 1, for the output
            to pass. A probability equal to the threshold passes.
        client: A TypeSafe client, or any object with a compatible ``system_one``
            method. When omitted, a ``typesafe_sdk.TypeSafeClient`` is created for
            each check (reading ``TYPESAFE_API_KEY`` and the other ``TYPESAFE_*``
            settings) and closed afterwards.
        model: Optional Jev model override. ``None`` uses the client's default.
        question_name: Name of the question in the request and response.

    Returns:
        A guardrail function for ``Task(guardrail=...)`` or ``guardrails=[...]``.
        It returns ``(True, output.raw)`` on success and ``(False, reason)`` on
        rejection.

    Raises:
        ValueError: If the question, question name or threshold is invalid.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(question_name, str) or not question_name.strip():
        raise ValueError("question_name must be a non-empty string")
    if not _is_probability(threshold):
        raise ValueError(
            f"threshold must be a finite number from 0 to 1, got {threshold!r}"
        )
    min_probability = float(threshold)

    def jev_noul_guardrail(output: TaskOutput | LiteAgentOutput) -> tuple[bool, Any]:
        questions = {question_name: {"type": "noul", "instructions": question}}
        if client is not None:
            response = client.system_one(
                state=output.raw, questions=questions, model=model
            )
        else:
            default_client = _create_default_client()
            try:
                response = default_client.system_one(
                    state=output.raw, questions=questions, model=model
                )
            finally:
                default_client.close()

        probability, problem = _read_noul_probability(response, question_name)
        if probability is None:
            return (
                False,
                f"Jev Noul guardrail could not verify the output: {problem}",
            )
        if probability >= min_probability:
            return True, output.raw
        return (
            False,
            f"Jev Noul guardrail rejected the output: the probability that "
            f"{question!r} is true is {probability:.3f}, below the required "
            f"threshold of {min_probability:g}.",
        )

    return jev_noul_guardrail


def _is_probability(value: object) -> TypeIs[int | float]:
    """Check that ``value`` is a real, finite number from 0 to 1 (not a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0.0 <= value <= 1.0


def _read_noul_probability(
    response: Any, question_name: str
) -> tuple[float | None, str]:
    """Return the Noul probability for ``question_name``, or ``None`` and the reason."""
    answers = getattr(response, "answers", None)
    if not isinstance(answers, Mapping) or question_name not in answers:
        return None, f"the response has no answer named {question_name!r}."
    answer = answers[question_name]
    if getattr(answer, "type", None) != "noul":
        return None, f"the answer named {question_name!r} is not a Noul answer."
    value = getattr(answer, "noul", None)
    if not _is_probability(value):
        return (
            None,
            f"the Noul answer {value!r} is not a finite probability from 0 to 1.",
        )
    return float(value), ""


def _create_default_client() -> Any:
    """Create a ``typesafe_sdk.TypeSafeClient`` from the environment."""
    try:
        sdk = importlib.import_module("typesafe_sdk")
    except ImportError as e:
        raise ImportError(
            "The 'typesafe-sdk' package is required for the Jev Noul guardrail "
            "when no client is passed. Install it with: uv add typesafe-sdk"
        ) from e
    return sdk.TypeSafeClient()
