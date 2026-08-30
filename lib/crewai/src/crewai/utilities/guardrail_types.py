"""Type aliases for guardrails."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

from crewai.lite_agent_output import LiteAgentOutput
from crewai.tasks.task_output import TaskOutput


GuardrailCallable: TypeAlias = Callable[
    [TaskOutput | LiteAgentOutput], tuple[bool, Any]
]

GuardrailType: TypeAlias = GuardrailCallable | str

GuardrailsType: TypeAlias = Sequence[GuardrailType] | GuardrailType


class GuardrailExecutionError(Exception):
    """The guardrail could not run. Not a statement about the output.

    Raised when the validation itself fails (e.g. the LLM call behind an
    ``LLMGuardrail`` raises), as opposed to the guardrail running and judging
    the output invalid. Callers must not treat it as a validation verdict:
    it does not count against ``guardrail_max_retries`` and the error text
    must not be fed back into the agent's conversation.
    """
