"""Hallucination Guardrail Placeholder for CrewAI.

This is a no-op version of the HallucinationGuardrail for the open-source repository.

Classes:
    HallucinationGuardrail: Placeholder guardrail that validates task outputs.
"""

from collections.abc import Callable
from typing import Any

from crewai.llm import LLM
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.logger import Logger


_validate_output_hook: Callable[..., tuple[bool, Any]] | None = None


class HallucinationGuardrail:
    """Placeholder for the HallucinationGuardrail feature.

    Attributes:
        context: Optional reference context that outputs would be checked against.
        llm: The language model that would be used for evaluation.
        threshold: Optional minimum faithfulness score that would be required to pass.
        tool_response: Optional tool response information that would be used in evaluation.

    Examples:
        >>> # Basic usage without context (uses task expected_output as context)
        >>> guardrail = HallucinationGuardrail(llm=agent.llm)

        >>> # With context for reference
        >>> guardrail = HallucinationGuardrail(
        ...     llm=agent.llm,
        ...     context="AI helps with various tasks including analysis and generation.",
        ... )

        >>> # With custom threshold for stricter validation
        >>> strict_guardrail = HallucinationGuardrail(
        ...     llm=agent.llm,
        ...     threshold=8.0,  # Require score >= 8 to pass
        ... )

        >>> # With tool response for additional context
        >>> guardrail_with_tools = HallucinationGuardrail(
        ...     llm=agent.llm,
        ...     tool_response="Weather API returned: Temperature 22°C, Humidity 65%",
        ... )
    """

    def __init__(
        self,
        llm: LLM,
        context: str | None = None,
        threshold: float | None = None,
        tool_response: str = "",
    ):
        """Initialize the HallucinationGuardrail placeholder.

        Args:
            llm: The language model that would be used for evaluation.
            context: Optional reference context that outputs would be checked against.
                If not provided, the task's expected_output will be used as context.
            threshold: Optional minimum faithfulness score that would be required to pass.
            tool_response: Optional tool response information that would be used in evaluation.
        """
        self.context = context
        self.llm: LLM = llm
        self.threshold = threshold
        self.tool_response = tool_response
        self._logger = Logger(verbose=True)
        self._logger.log(
            "warning",
            """Hallucination detection is a no-op in open source, use it for free at https://app.crewai.com\n""",
            color="red",
        )

    @property
    def description(self) -> str:
        """Generate a description of this guardrail for event logging."""
        return "HallucinationGuardrail (no-op)"

    def __call__(self, task_output: TaskOutput) -> tuple[bool, Any]:
        """Validate a task output against hallucination criteria.

        Args:
            task_output: The output to be validated.

        Returns:
            A tuple containing:
                - True if validation passed, False otherwise
                - The raw task output if valid, or error feedback if invalid
        """
        if callable(_validate_output_hook):
            return _validate_output_hook(self, task_output)

        self._logger.log(
            "warning",
            "Premium hallucination detection skipped (use for free at https://app.crewai.com)\n",
            color="red",
        )
        return True, task_output.raw
